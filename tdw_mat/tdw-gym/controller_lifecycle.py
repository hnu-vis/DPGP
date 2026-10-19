"""Best-effort lifecycle helpers for TDW controllers."""


def shutdown_controller(controller):
    """Terminate a TDW build and release its Python-side socket.

    Returns a list of cleanup errors instead of masking the original episode
    exception. Socket closure is always attempted, even if TDW no longer replies
    to the terminate command.
    """
    if controller is None:
        return []

    errors = []
    try:
        controller.communicate({"$type": "terminate"})
    except Exception as error:  # Cleanup must continue after a dead build.
        errors.append(("terminate", error))

    socket = getattr(controller, "socket", None)
    if socket is not None:
        try:
            socket.close(linger=0)
        except TypeError:
            # Keep compatibility with simple/fake sockets and older pyzmq.
            try:
                socket.close()
            except Exception as error:  # pragma: no cover - defensive fallback
                errors.append(("socket_close", error))
        except Exception as error:
            errors.append(("socket_close", error))
    return errors
