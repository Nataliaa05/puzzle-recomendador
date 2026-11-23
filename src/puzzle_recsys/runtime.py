"""Utilidades de ejecución local.

Las corridas largas (simulación, Optuna) duran más que el tiempo de suspensión de
una laptop. Si Windows se suspende, el kernel del notebook muere y se pierde el
trabajo. ``keep_awake`` le pide al sistema no suspenderse mientras el proceso
siga vivo (no evita la suspensión al cerrar la tapa).

Usado en: notebooks 12.
"""

from __future__ import annotations

import sys

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake() -> bool:
    """Evita la suspensión automática mientras corre el proceso; ``True`` si se pudo."""
    if sys.platform != "win32":
        return False
    import ctypes

    return bool(
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    )
