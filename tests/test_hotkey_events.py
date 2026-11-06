import importlib.util
import logging
import os
import unittest
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(
    os.environ.get("WHISPER_SOURCE_ROOT", Path(__file__).resolve().parents[1])
)
HELPER_PATH = ROOT / "src" / "hotkey_events.py"
MAIN_PATH = ROOT / "src" / "main.py"


def load_hotkey_helper():
    if not HELPER_PATH.exists():
        raise AssertionError(
            "global-listener diagnostics must use the configured-hotkey boundary"
        )
    spec = importlib.util.spec_from_file_location("hotkey_events", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.log_configured_hotkey_event


class Key:
    def __init__(self, vk=None):
        self.vk = vk


class HotkeyEventLoggingTests(unittest.TestCase):
    def test_release_callback_has_no_unfiltered_keycode_log(self):
        source = MAIN_PATH.read_text(encoding="utf-8")

        self.assertNotIn(
            'logger.debug(f"Key with vk={key.vk} released")',
            source,
        )

    def test_unrelated_global_key_is_not_logged(self):
        logger = Mock(spec=logging.Logger)
        log_configured_hotkey_event = load_hotkey_helper()

        matched = log_configured_hotkey_event(
            logger, Key(vk=12), trigger_key=63, action="released"
        )

        self.assertFalse(matched)
        logger.debug.assert_not_called()

    def test_configured_hotkey_keeps_diagnostic_and_matches(self):
        logger = Mock(spec=logging.Logger)
        log_configured_hotkey_event = load_hotkey_helper()

        matched = log_configured_hotkey_event(
            logger, Key(vk=63), trigger_key=63, action="released"
        )

        self.assertTrue(matched)
        logger.debug.assert_called_once_with(
            "Target key (vk=%s) %s", 63, "released"
        )

    def test_key_without_virtual_keycode_is_not_logged(self):
        logger = Mock(spec=logging.Logger)
        log_configured_hotkey_event = load_hotkey_helper()

        matched = log_configured_hotkey_event(
            logger, object(), trigger_key=63, action="released"
        )

        self.assertFalse(matched)
        logger.debug.assert_not_called()


if __name__ == "__main__":
    unittest.main()
