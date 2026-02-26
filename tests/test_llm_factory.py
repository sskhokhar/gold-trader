from __future__ import annotations

import os
import unittest

from gold_trading_one_trade_per_day.llm_factory import (
    load_llm_runtime_config,
    resolve_llm_route,
)


class TestLLMFactory(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_defaults_load(self):
        os.environ.pop("AGENT_MODEL_ANALYST", None)
        os.environ.pop("AGENT_MODEL_STRATEGY", None)
        cfg = load_llm_runtime_config()
        self.assertEqual(cfg.analyst_model, "ollama/llama4:8b")
        self.assertEqual(cfg.strategy_model, "gemini/gemini-2.0-flash")
        self.assertEqual(cfg.max_retries, 5)

    def test_shadow_falls_back_to_gemini_when_local_unavailable(self):
        os.environ["AGENT_MODEL_ANALYST"] = "ollama/llama4:8b"
        os.environ["AGENT_MODEL_STRATEGY"] = "gemini/gemini-2.0-flash"
        os.environ["ALLOW_ANALYST_GEMINI_FALLBACK_SHADOW"] = "true"
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:9"
        cfg = load_llm_runtime_config()
        route = resolve_llm_route(mode="shadow", cfg=cfg)
        self.assertEqual(route.analyst_model, "gemini/gemini-2.0-flash")
        self.assertTrue(route.analyst_fallback_used)

    def test_paper_uses_local_strategy_when_not_explicitly_set(self):
        os.environ["AGENT_MODEL_ANALYST"] = "ollama/llama3.1:8b"
        os.environ.pop("AGENT_MODEL_STRATEGY", None)
        cfg = load_llm_runtime_config()
        route = resolve_llm_route(mode="paper", cfg=cfg)
        self.assertEqual(route.strategy_model, "ollama/llama3.1:8b")
        self.assertTrue(route.strategy_auto_overridden)

    def test_paper_keeps_strategy_when_explicitly_set(self):
        os.environ["AGENT_MODEL_ANALYST"] = "ollama/llama3.1:8b"
        os.environ["AGENT_MODEL_STRATEGY"] = "gemini/gemini-2.0-flash"
        cfg = load_llm_runtime_config()
        route = resolve_llm_route(mode="paper", cfg=cfg)
        self.assertEqual(route.strategy_model, "ollama/llama3.1:8b")
        self.assertTrue(route.strategy_auto_overridden)

    def test_paper_can_opt_out_of_local_strategy_override(self):
        os.environ["AGENT_MODEL_ANALYST"] = "ollama/llama3.1:8b"
        os.environ["AGENT_MODEL_STRATEGY"] = "gemini/gemini-2.0-flash"
        os.environ["PAPER_USE_LOCAL_STRATEGY"] = "false"
        cfg = load_llm_runtime_config()
        route = resolve_llm_route(mode="paper", cfg=cfg)
        self.assertEqual(route.strategy_model, "gemini/gemini-2.0-flash")
        self.assertFalse(route.strategy_auto_overridden)


if __name__ == "__main__":
    unittest.main()
