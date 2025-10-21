import logging
import random
import time
from collections import defaultdict

import pytest
from inference_endpoint import metrics
from inference_endpoint.config.ruleset import RuntimeSettings
from inference_endpoint.dataset_manager.dataloader import (
    DataLoader,
    DeepSeekR1ChatCompletionDataLoader,
)
from inference_endpoint.endpoint_client.configs import (
    AioHttpConfig,
    HTTPClientConfig,
    ZMQConfig,
)
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.loadgen import HttpClientSampleIssuer
from inference_endpoint.load_generator import LoadGenerator
from inference_endpoint.load_generator.scheduler import (
    MaxThroughputScheduler,
    SampleFactory,
    WithoutReplacementSampleOrder,
)

from tests.test_helpers import get_test_socket_path


class DeepSeekR1SampleFactory(SampleFactory):
    _output_histogram = defaultdict(int)
    request_data: {str: str} = {}

    @staticmethod
    def sample_get_bytes(dataloader: DataLoader, sample_index: int):
        logging.info(f"Sample get bytes: {sample_index}")
        return dataloader.load_sample(sample_index)

    @staticmethod
    def sample_complete_callback(output, sid):
        logging.info(f"Sample complete callback: {output}")
        DeepSeekR1SampleFactory.request_data[sid] = output


class DeepSeekR1SampleIssuer(HttpClientSampleIssuer):
    def __init__(self, tmp_path: str, url: str):
        self.http_config = HTTPClientConfig(
            endpoint_url=f"{url}/v1/chat/completions",
            num_workers=2,
            max_concurrency=10,
        )
        self.aiohttp_config = AioHttpConfig()
        self.zmq_config = ZMQConfig(
            zmq_request_queue_prefix=get_test_socket_path(
                tmp_path, "test_streaming", "_req"
            ),
            zmq_response_queue_addr=get_test_socket_path(
                tmp_path, "test_streaming", "_resp"
            ),
            zmq_readiness_queue_addr=get_test_socket_path(
                tmp_path, "test_streaming", "_ready"
            ),
        )
        super().__init__(
            HTTPEndpointClient(self.http_config, self.aiohttp_config, self.zmq_config)
        )


@pytest.mark.asyncio
async def test_load_generator_full_run(
    mock_http_oracle_server, ds_pickle_dataset_path, tmp_path
):
    DeepSeekR1SampleFactory._output_histogram.clear()

    def parser(x):
        return {"prompt": x.text_input, "output": x.ref_output}

    dummy_dataloader = DeepSeekR1ChatCompletionDataLoader(
        ds_pickle_dataset_path, parser
    )
    dummy_dataloader.load()
    assert dummy_dataloader.num_samples() > 0

    rt_settings = RuntimeSettings(
        metrics.Throughput(5000),
        [metrics.Throughput(5000)],
        min_duration_ms=1_000,
        max_duration_ms=10_000_000,
        n_samples_from_dataset=dummy_dataloader.num_samples(),
        n_samples_to_issue=dummy_dataloader.num_samples(),
        rng_sched=random.Random(1234),
        rng_sample_index=random.Random(1234),
    )

    scheduler = MaxThroughputScheduler(
        rt_settings,
        dummy_dataloader,
        DeepSeekR1SampleFactory,
        WithoutReplacementSampleOrder,
    )
    logging.info(f"Number of samples to issue: {scheduler.total_samples_to_issue}")

    try:
        sample_issuer = DeepSeekR1SampleIssuer(tmp_path, mock_http_oracle_server.url)
        # Start HTTP client first to initialize its event loop
        sample_issuer.http_client.start()
        # Then start the sample issuer which needs the client's loop
        sample_issuer.start()
        load_generator = LoadGenerator(scheduler, sample_issuer)
        sess = load_generator.start_test()

        sess.wait_for_test_end()
        end_time_ns = time.monotonic_ns()
    except Exception as e:
        logging.info(f"Error: {e}")
        raise e
    finally:
        logging.info(f"Request data: {DeepSeekR1SampleFactory.request_data}")

    assert sess.start_time_ns is not None
    assert end_time_ns is not None
    try:
        vals = {}
        for i in range(dummy_dataloader.num_samples()):
            entry = dummy_dataloader.load_sample(i)
            vals[i] = entry["output"]
        num_samples_in_dataset = len(vals)

        logging.info(f"Number of samples in dataset: {num_samples_in_dataset}")
        logging.info(f"Total samples to issue: {scheduler.total_samples_to_issue}")
        logging.info(f"Request data: {len(DeepSeekR1SampleFactory.request_data)}")
        assert (
            num_samples_in_dataset == scheduler.total_samples_to_issue
        ), "Number of samples in dataset and number of samples in request data should be the same"

        for req, resp in DeepSeekR1SampleFactory.request_data.items():
            response = resp.response_output
            logging.info(
                f"Sample {req} should have been response {vals[req][0:30]}, but was response {response[0:30]}"
            )
            assert (
                response == vals[req]
            ), f"Sample {req} should have been response {vals[req][0:30]}, but was response {response[0:30]}"
    except Exception as e:
        logging.info(f"Error: {e}")
        raise e
    finally:
        # Cleanup: shutdown the sample issuer and HTTP client
        sample_issuer.shutdown()
        sample_issuer.http_client.shutdown()
