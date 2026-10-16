"""Keep the gripper training from starving the rest of the laptop (e.g. a browser holding a Colab tab).

free_gb() reads free physical RAM (psutil if installed, else Windows' GlobalMemoryStatusEx, else
/proc/meminfo). wait_for_ram() blocks while it is below GRIPPER_MIN_FREE_GB (default 1 GB), checking
every minute; RamGuard does the same between PPO rollouts, so training pauses instead of pushing the
machine into swap, and carries on by itself once memory is free again.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time

from stable_baselines3.common.callbacks import BaseCallback

MIN_FREE_GB = float(os.environ.get("GRIPPER_MIN_FREE_GB", "1.0"))


def free_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except ImportError:
        pass
    if sys.platform == "win32":
        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MemStatus()
        m.dwLength = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m.ullAvailPhys / 2**30
        return None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def wait_for_ram(min_gb: float = MIN_FREE_GB) -> None:
    free = free_gb()
    while free is not None and free < min_gb:
        print(f"ram guard: {free:.1f} GB free < {min_gb} GB, pausing 60 s", flush=True)
        time.sleep(60)
        free = free_gb()


class RamGuard(BaseCallback):
    def _on_rollout_start(self) -> None:
        wait_for_ram()

    def _on_step(self) -> bool:
        return True
