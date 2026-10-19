"""Scene-command helpers for optional diagnostic cameras."""

TOP_DOWN_AVATAR_ID = "a"


def filter_top_down_camera_commands(commands, enabled):
    """Return scene commands with the diagnostic top-down avatar optionally removed.

    The dataset camera is independent of Replicant avatars and isn't part of an
    agent observation. Filtering its commands avoids rendering and transferring
    an unused image stream when top-down recording is disabled.
    """
    if enabled:
        return list(commands)
    return [
        command
        for command in commands
        if not (
            (
                command.get("$type") == "create_avatar"
                and command.get("id") == TOP_DOWN_AVATAR_ID
            )
            or command.get("avatar_id") == TOP_DOWN_AVATAR_ID
        )
    ]
