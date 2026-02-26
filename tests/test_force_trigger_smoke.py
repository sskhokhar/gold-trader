from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from gold_trading_one_trade_per_day import main as main_mod


class TestForceTriggerSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_live_mode_ignores_force_flag_when_live_disabled(self):
        os.environ["ENABLE_LIVE_TRADING"] = "false"
        os.environ["FORCE_TRIGGER_ONCE"] = "true"
        result = main_mod._run_cycle("live")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "live_mode_disabled")
        self.assertTrue(result.get("force_trigger_ignored_live"))

    def test_run_with_trigger_aliases_smoke(self):
        with patch.object(main_mod, "run_paper_smoke") as mock_smoke:
            main_mod.run_with_trigger()
        mock_smoke.assert_called_once_with()

    def test_shadow_loop_consumes_force_once_after_first_applied(self):
        os.environ["FORCE_TRIGGER_ONCE"] = "true"
        os.environ["SHADOW_LOOP_SLEEP_SEC"] = "0"
        calls: list[bool] = []
        iteration = {"value": 0}

        def fake_run_cycle(mode: str, force_trigger_once: bool = False, force_trigger_source: str = "env"):
            self.assertEqual(mode, "shadow")
            self.assertEqual(force_trigger_source, "env")
            calls.append(force_trigger_once)
            iteration["value"] += 1
            # Consume force on 2nd iteration to simulate "first eligible skipped event".
            return {"force_trigger_applied": iteration["value"] == 2}

        with patch.object(main_mod, "_commandless_args", return_value=["3"]), patch.object(
            main_mod, "_run_cycle", side_effect=fake_run_cycle
        ), patch.object(main_mod.time_module, "sleep", return_value=None), patch(
            "builtins.print"
        ):
            main_mod.run_shadow_loop()

        self.assertEqual(calls, [True, True, False])


if __name__ == "__main__":
    unittest.main()
