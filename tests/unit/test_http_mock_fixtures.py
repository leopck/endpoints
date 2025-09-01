"""
Tests demonstrating the usage of HTTP echo mock fixtures.

These tests show how to use the mock fixtures for testing HTTP clients
with real HTTP server that echoes requests back.
"""

import aiohttp
import pytest
from inference_endpoint.core.types import ChatCompletionQuery


class TestHttpEchoMockFixtures:
    """Test suite demonstrating HTTP mock fixture usage."""

    @pytest.mark.asyncio
    async def test_mock_http_echo_server_basic(self, mock_http_echo_server):
        """Test basic echo functionality of the real HTTP server."""
        server = mock_http_echo_server

        # Make a real HTTP request to the server
        async with aiohttp.ClientSession() as session:
            payload = {"prompt": "Hello, world!", "model": "gpt-3.5-turbo"}
            async with session.post(
                f"{server.url}/v1/completions", json=payload
            ) as response:
                assert response.status == 200

                response_data = await response.json()

                assert response_data["echo"] is True
                assert response_data["request"]["method"] == "POST"
                assert response_data["request"]["endpoint"] == "/v1/completions"
                assert (
                    response_data["request"]["json_payload"]["prompt"]
                    == "Hello, world!"
                )
                assert (
                    response_data["request"]["json_payload"]["model"] == "gpt-3.5-turbo"
                )

    @pytest.mark.asyncio
    async def test_mock_http_echo_server_chat_completions(self, mock_http_echo_server):
        """Test basic echo functionality of the real HTTP server."""
        server = mock_http_echo_server

        # Make a real HTTP OpenAI chat completions request to the server
        async with aiohttp.ClientSession() as session:
            prompt_text = "Test prompt for mock server"
            payload = ChatCompletionQuery(
                prompt=prompt_text, model="gpt-3.5-turbo"
            ).to_json()
            async with session.post(
                f"{server.url}/v1/chat/completions", json=payload
            ) as response:
                assert response.status == 200

                response_data = await response.json()

                assert response_data["echo"] is True
                assert response_data["request"]["method"] == "POST"
                assert response_data["request"]["endpoint"] == "/v1/chat/completions"
                assert (
                    response_data["request"]["json_payload"]["messages"][1]["content"]
                    == prompt_text
                )
                assert (
                    response_data["request"]["json_payload"]["model"] == "gpt-3.5-turbo"
                )

    @pytest.mark.asyncio
    async def test_real_http_server_post_request(self, mock_http_echo_server):
        """Test POST request to real HTTP server."""
        server = mock_http_echo_server

        async with aiohttp.ClientSession() as session:
            payload = {
                "query": "What is machine learning?",
                "parameters": {"temperature": 0.7, "max_tokens": 150},
            }

            async with session.post(
                f"{server.url}/v1/completions", json=payload
            ) as response:
                assert response.status == 200

                response_data = await response.json()

                # Verify echo response structure
                assert response_data["echo"] is True
                assert response_data["request"]["method"] == "POST"
                assert response_data["request"]["endpoint"] == "/v1/completions"
                assert response_data["request"]["json_payload"] == payload
