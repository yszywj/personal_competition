"""Strict response parsing and JSON-mode transport fallback tests."""

from __future__ import annotations

import json
import unittest

from personal_train.llm_strategy.glm_client import (
    GLMClient,
    LLMClientError,
    extract_json_payload,
)


def completion(content: str = '{"plan_version":"v0"}') -> str:
    return json.dumps(
        {
            "model": "glm-test",
            "choices": [
                {"message": {"content": content}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    )


class StubGLMClient(GLMClient):
    def __init__(self, responses, *, json_mode="auto"):
        super().__init__(
            api_key="test-key",
            base_url="https://example.invalid",
            model="glm-test",
            json_mode=json_mode,
        )
        self.responses = list(responses)
        self.posted_payloads = []

    def _post(self, payload):
        self.posted_payloads.append(payload)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class StrictJSONParsingTests(unittest.TestCase):
    def test_bare_json_object_is_accepted(self):
        payload, error = extract_json_payload('{"plan_version":"v0"}')
        self.assertIsNone(error)
        self.assertEqual(payload, {"plan_version": "v0"})

    def test_complete_json_fence_is_accepted(self):
        fence = chr(96) * 3
        payload, error = extract_json_payload(
            fence + 'json\n{"plan_version":"v0"}\n' + fence
        )
        self.assertIsNone(error)
        self.assertEqual(payload, {"plan_version": "v0"})

    def test_natural_language_before_json_is_rejected(self):
        payload, error = extract_json_payload(
            'Here is the plan: {"plan_version":"v0"}'
        )
        self.assertIsNone(payload)
        self.assertIsNotNone(error)

    def test_natural_language_after_json_is_rejected(self):
        payload, error = extract_json_payload(
            '{"plan_version":"v0"} This is the plan.'
        )
        self.assertIsNone(payload)
        self.assertIsNotNone(error)

    def test_malformed_json_is_rejected(self):
        payload, error = extract_json_payload('{"plan_version":"v0",}')
        self.assertIsNone(payload)
        self.assertIsNotNone(error)

    def test_incomplete_or_unlabelled_fence_is_rejected(self):
        fence = chr(96) * 3
        cases = (
            fence + 'json\n{"plan_version":"v0"}',
            fence + '\n{"plan_version":"v0"}\n' + fence,
            fence + 'json\n{"plan_version":"v0"}\n' + fence + ' trailing',
        )
        for raw in cases:
            with self.subTest(raw=raw):
                payload, error = extract_json_payload(raw)
                self.assertIsNone(payload)
                self.assertIsNotNone(error)


class JSONModeFallbackTests(unittest.TestCase):
    def test_explicit_unsupported_parameter_falls_back_once(self):
        client = StubGLMClient(
            [
                (400, '{"error":"response_format json_object is not supported"}'),
                (200, completion()),
            ]
        )
        response = client.chat("system", "user")
        self.assertEqual(client.call_count, 1)
        self.assertEqual(response.http_request_count, 2)
        self.assertTrue(response.json_mode_requested)
        self.assertTrue(response.json_mode_fallback)
        self.assertIn("response_format", client.posted_payloads[0])
        self.assertNotIn("response_format", client.posted_payloads[1])

    def test_4xx_without_explicit_unsupported_semantics_does_not_fallback(self):
        client = StubGLMClient(
            [(400, '{"error":"response_format request failed"}')]
        )
        with self.assertRaises(LLMClientError):
            client.chat("system", "user")
        self.assertEqual(len(client.posted_payloads), 1)

    def test_server_error_does_not_fallback(self):
        client = StubGLMClient(
            [(500, '{"error":"response_format is not supported"}')]
        )
        with self.assertRaises(LLMClientError):
            client.chat("system", "user")
        self.assertEqual(len(client.posted_payloads), 1)

    def test_rate_limit_does_not_fallback(self):
        client = StubGLMClient(
            [(429, '{"error":"response_format is not supported; rate limit"}')]
        )
        with self.assertRaises(LLMClientError):
            client.chat("system", "user")
        self.assertEqual(len(client.posted_payloads), 1)

    def test_timeout_does_not_fallback(self):
        client = StubGLMClient([LLMClientError("GLM request failed: timeout")])
        with self.assertRaises(LLMClientError):
            client.chat("system", "user")
        self.assertEqual(len(client.posted_payloads), 1)

    def test_invalid_generated_content_does_not_trigger_transport_fallback(self):
        client = StubGLMClient([(200, completion("not json"))])
        response = client.chat("system", "user")
        payload, error = extract_json_payload(response.content)
        self.assertIsNone(payload)
        self.assertIsNotNone(error)
        self.assertEqual(client.call_count, 1)
        self.assertEqual(response.http_request_count, 1)
        self.assertFalse(response.json_mode_fallback)
        self.assertEqual(len(client.posted_payloads), 1)


if __name__ == "__main__":
    unittest.main()
