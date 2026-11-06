"""Helpers for safely diagnosing events from the global keyboard listener."""


def log_configured_hotkey_event(logger, key, trigger_key, action):
    """Log an event only when it belongs to the configured application hotkey.

    The listener receives events from every application, so unrelated key
    details must not cross the diagnostic boundary.
    """
    if getattr(key, "vk", None) != trigger_key:
        return False

    logger.debug("Target key (vk=%s) %s", trigger_key, action)
    return True
