"""Detect the host's storage topology and judge whether a path is SAFE for a given run role.

This is the read side of §11/§20 of the platform vision: the run planner needs to assign the
checkpoint / scratch / optimizer-offload / parameter-offload paths, and putting an offload file on a
USB bridge, a cloud-sync folder, a nearly-full disk, or inside the source repository is a
train-halting (or data-losing) mistake. This module characterizes the available storage with a
*dependency-light, non-destructive* probe and produces a per-role suitability verdict.

Deliberately NOT here (a later, consent-gated slice): measured sequential/random throughput, a bounded
write benchmark, and SMART/NVMe endurance (`data_units_written`, TBW, temperature). Those need either
a real benchmark or a privileged device read. Everything this module can't cheaply determine stays
``None``/``unknown`` — an honest gap, never a guessed number.

The suitability logic (:func:`assess_role`) is PURE over an injected :class:`StorageDevice`, so every
verdict is unit-tested without touching the real host; the detection functions wrap it with the
platform-specific I/O.
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .contracts import StorageDevice, StorageProfile, StorageRoleAssessment
from .enums import StorageInterface, StorageRole, StorageSuitability
from .host_platform import is_wsl

# Roles whose write pattern is sustained + heavy — the ones a slow/removable/synced device ruins.
_HIGH_WRITE_ROLES: frozenset[StorageRole] = frozenset(
    {
        StorageRole.checkpoints,
        StorageRole.scratch,
        StorageRole.optimizer_offload,
        StorageRole.parameter_offload,
    }
)
# Roles that must never live inside the source repository (they'd pollute it / risk being committed).
_NO_SOURCE_REPO_ROLES: frozenset[StorageRole] = _HIGH_WRITE_ROLES | frozenset({StorageRole.artifacts})
# The offload roles specifically — where a rotational disk is merely slow (marginal), not forbidden.
_OFFLOAD_ROLES: frozenset[StorageRole] = frozenset(
    {StorageRole.optimizer_offload, StorageRole.parameter_offload}
)

_GB = 1_000_000_000
# Advisory minimum free-space margins per role (bytes). Heuristic, documented as such — a floor to
# catch a nearly-full disk, not a precise per-run requirement (the planner refines with a real
# estimate). None = no minimum (a role that writes little).
_MIN_FREE_BYTES: dict[StorageRole, int] = {
    StorageRole.optimizer_offload: 20 * _GB,
    StorageRole.parameter_offload: 20 * _GB,
    StorageRole.checkpoints: 20 * _GB,
    StorageRole.scratch: 10 * _GB,
    StorageRole.model_cache: 30 * _GB,
    StorageRole.dataset_cache: 5 * _GB,
    StorageRole.artifacts: 10 * _GB,
    StorageRole.archive: 1 * _GB,
    StorageRole.os: 5 * _GB,
    StorageRole.source_repo: 2 * _GB,
    StorageRole.logs: 1 * _GB,
}

# Cloud-sync client folder names. A path under one of these means a sync daemon re-uploads every
# checkpoint/offload write — thrashing the disk + the network. Specific names only (no bare "Box"/
# "Sync") to avoid false positives; OneDrive is matched by prefix ("OneDrive - Company").
_CLOUD_SYNC_EXACT: frozenset[str] = frozenset(
    {
        "dropbox",
        "google drive",
        "googledrive",
        "icloud drive",
        "icclouddrive",
        "iclouddrive",
        "pcloud",
        "nextcloud",
        "owncloud",
        "mega",
        "megasync",
        "creative cloud files",
    }
)
_CLOUD_SYNC_PREFIXES: tuple[str, ...] = ("onedrive",)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------------------------------
# Pure classifiers + suitability logic (fully unit-testable without touching the host).
# --------------------------------------------------------------------------------------------------
def classify_interface(
    device_name: str | None, *, removable: bool | None, rotational: bool | None
) -> StorageInterface:
    """Best-effort interface from the cheaply-discoverable device attributes. Removable wins (a
    removable NVMe in a USB enclosure is still a USB bridge for our purposes); then an ``nvme`` device
    name; then rotational splits HDD vs SATA SSD. ``unknown`` when nothing is determinable."""
    if removable:
        return StorageInterface.usb
    if device_name and os.path.basename(device_name).lower().startswith("nvme"):
        return StorageInterface.nvme_pcie
    if rotational is True:
        return StorageInterface.hdd
    if rotational is False:
        return StorageInterface.sata_ssd
    return StorageInterface.unknown


def is_cloud_synced_path(path: str) -> bool:
    """True when any component of ``path`` is a known cloud-sync client folder. Splits on BOTH path
    separators explicitly (not ``Path().parts``) so a Windows path is checked correctly even on a
    POSIX host — the storage profile of one OS is sometimes read on another (WSL, tests, remote)."""
    for part in path.replace("\\", "/").split("/"):
        low = part.strip().lower()
        if low in _CLOUD_SYNC_EXACT or any(low.startswith(p) for p in _CLOUD_SYNC_PREFIXES):
            return True
    return False


def is_inside_source_repo(path: str) -> bool:
    """True when ``path`` is inside a git working tree (a ``.git`` exists at it or an ancestor).
    Best-effort filesystem walk; any error → False (never crash a probe)."""
    try:
        current = Path(path).resolve()
    except (OSError, ValueError):
        return False
    for candidate in (current, *current.parents):
        try:
            if (candidate / ".git").exists():
                return True
        except OSError:
            return False
    return False


def min_free_bytes_for_role(role: StorageRole) -> int | None:
    """The advisory minimum free-space margin for a role, or None when the role writes little."""
    return _MIN_FREE_BYTES.get(role)


def assess_role(
    *,
    role: StorageRole,
    path: str,
    device: StorageDevice | None,
    inside_source_repo: bool,
) -> StorageRoleAssessment:
    """The PURE per-role suitability verdict for a candidate ``path`` on a characterized ``device``.

    A single ``unsuitable`` reason (data-loss or thrash-to-a-halt risk) makes the whole verdict
    ``unsuitable``; otherwise any ``marginal`` reason (works but degraded) wins over ``suitable``. When
    the device couldn't be characterized the verdict is ``unknown`` — never a false ``suitable``.
    """
    required = min_free_bytes_for_role(role)
    if device is None:
        return StorageRoleAssessment(
            role=role,
            path=path,
            suitability=StorageSuitability.unknown,
            required_free_bytes=required,
            reasons=["could not characterize the storage device at this path"],
        )

    unsuitable: list[str] = []
    marginal: list[str] = []

    # Cloud-sync folder: a sync client re-uploads every write. Fatal for anything that writes a lot.
    if device.cloud_synced and (role in _HIGH_WRITE_ROLES or role == StorageRole.artifacts):
        unsuitable.append(
            "inside a cloud-sync folder - a sync client will re-upload every write and thrash the disk"
        )

    # Generated high-write / artifact state must not live inside the source repository.
    if inside_source_repo and role in _NO_SOURCE_REPO_ROLES:
        unsuitable.append(
            "inside the source repository - generated run state must not pollute (or be committed to) source"
        )

    # Interface: a USB bridge or network mount cannot sustain offload/checkpoint write traffic.
    if role in _HIGH_WRITE_ROLES:
        if device.interface == StorageInterface.usb:
            unsuitable.append(
                "on a USB/removable device - unfit for the sustained write traffic of offload/checkpointing"
            )
        elif device.interface == StorageInterface.network:
            unsuitable.append(
                "on a network mount - latency and reliability are unfit for offload/checkpointing"
            )

    # A rotational disk works for offload but is far slower than NVMe - a real but non-fatal cost.
    if role in _OFFLOAD_ROLES and (
        device.interface == StorageInterface.hdd or device.rotational is True
    ):
        marginal.append("on a rotational disk - offload will be I/O-bound and slow; prefer internal NVMe")

    # Free-space margin: a nearly-full disk will fail mid-run.
    if required is not None and device.free_bytes is not None and device.free_bytes < required:
        unsuitable.append(
            f"only {device.free_bytes / _GB:.1f} GB free - this role needs roughly "
            f"{required / _GB:.0f} GB of headroom"
        )

    if unsuitable:
        verdict = StorageSuitability.unsuitable
    elif marginal:
        verdict = StorageSuitability.marginal
    else:
        verdict = StorageSuitability.suitable

    return StorageRoleAssessment(
        role=role,
        path=path,
        suitability=verdict,
        device_mount_point=device.mount_point,
        interface=device.interface,
        free_bytes=device.free_bytes,
        required_free_bytes=required,
        reasons=[*unsuitable, *marginal],
    )


def parse_proc_mounts(text: str) -> list[tuple[str, str, str]]:
    """Parse ``/proc/mounts`` content into ``(device, mount_point, fstype)`` triples, keeping only
    real (non-pseudo) filesystems. Pure — fed fixture text in tests."""
    pseudo = {
        "proc",
        "sysfs",
        "devtmpfs",
        "devpts",
        "cgroup",
        "cgroup2",
        "securityfs",
        "pstore",
        "debugfs",
        "tracefs",
        "mqueue",
        "hugetlbfs",
        "bpf",
        "configfs",
        "fusectl",
        "binfmt_misc",
        "autofs",
        "ramfs",
    }
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        device, mount_point, fstype = fields[0], fields[1], fields[2]
        if fstype in pseudo:
            continue
        # Un-escape the octal mount-point encoding /proc/mounts uses for spaces (\040) etc.
        mount_point = mount_point.encode("utf-8").decode("unicode_escape")
        rows.append((device, mount_point, fstype))
    return rows


# --------------------------------------------------------------------------------------------------
# Platform-specific detection (best-effort I/O — never raises; honest "unknown" on any failure).
# --------------------------------------------------------------------------------------------------
def _disk_usage(mount_point: str) -> tuple[int | None, int | None]:
    try:
        usage = shutil.disk_usage(mount_point)
        return int(usage.total), int(usage.free)
    except (OSError, ValueError):
        return None, None


def _linux_block_flag(device: str, attribute: str) -> bool | None:
    """Read a 0/1 flag from ``/sys/block/<disk>/<attribute>`` (removable, queue/rotational). Returns
    None when it can't be read — a partition like ``/dev/nvme0n1p2`` maps back to its parent disk."""
    base = os.path.basename(device)
    if not base.startswith(("sd", "nvme", "vd", "hd", "mmcblk")):
        return None
    # Strip a partition suffix to reach the parent disk (nvme0n1p2 → nvme0n1, sda1 → sda).
    disk = base
    if base.startswith(("nvme", "mmcblk")):
        disk = base.split("p")[0] if "p" in base else base
    else:
        disk = base.rstrip("0123456789")
    try:
        with open(f"/sys/block/{disk}/{attribute}", encoding="ascii") as handle:
            return handle.read().strip() == "1"
    except OSError:
        return None


def _detect_linux_devices() -> list[StorageDevice]:
    """Detect mounted storage on Linux / WSL from /proc/mounts + /sys/block + disk_usage."""
    try:
        with open("/proc/mounts", encoding="utf-8") as handle:
            mounts = parse_proc_mounts(handle.read())
    except OSError:
        return _detect_generic_devices()

    wsl = is_wsl()
    devices: list[StorageDevice] = []
    seen: set[str] = set()
    for device_path, mount_point, fstype in mounts:
        if mount_point in seen:
            continue
        seen.add(mount_point)
        total, free = _disk_usage(mount_point)
        interface, removable, rotational, notes = classify_linux_mount(
            device_path, mount_point, fstype, wsl=wsl
        )
        devices.append(
            StorageDevice(
                mount_point=mount_point,
                filesystem=fstype,
                interface=interface,
                total_bytes=total,
                free_bytes=free,
                removable=removable,
                rotational=rotational,
                cloud_synced=is_cloud_synced_path(mount_point),
                device_name=device_path,
                notes=notes,
            )
        )
    return devices


def classify_linux_mount(
    device_path: str, mount_point: str, fstype: str, *, wsl: bool
) -> tuple[StorageInterface, bool | None, bool | None, list[str]]:
    """Classify one Linux/WSL mount into ``(interface, removable, rotational, notes)``. Pure w.r.t.
    the filesystem type + name; only the ``else`` (a real block device) reads ``/sys/block``. Split
    out so every branch — WSL host drive, network, virtual, real disk — is unit-tested."""
    notes: list[str] = []
    # Under WSL, /mnt/<drive> and drvfs/9p mounts are the Windows host drives seen through a
    # translation layer — the real device attributes aren't visible from Linux.
    if wsl and (fstype in {"drvfs", "9p", "v9fs"} or mount_point.startswith("/mnt/")):
        notes.append("WSL view of a Windows host drive - real device attributes not visible from Linux")
        return StorageInterface.virtual, None, None, notes
    if fstype in {"nfs", "nfs4", "cifs", "smb", "smbfs", "fuse.sshfs"}:
        return StorageInterface.network, None, None, notes
    if fstype in {"tmpfs", "overlay", "squashfs"}:
        return StorageInterface.virtual, None, None, notes
    removable = _linux_block_flag(device_path, "removable")
    rotational = _linux_block_flag(device_path, "queue/rotational")
    return classify_interface(device_path, removable=removable, rotational=rotational), removable, rotational, notes


def _detect_windows_devices() -> list[StorageDevice]:  # pragma: no cover - Windows-only; exercised on a real Windows host, not the Linux CI coverage runner.
    """Detect mounted volumes on Windows via GetDriveType + GetVolumeInformation + disk_usage."""
    try:
        import ctypes  # noqa: PLC0415 - Windows-only, imported lazily.

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return _detect_generic_devices()

    # DRIVE_* return codes from GetDriveTypeW (3 = DRIVE_FIXED, handled by the else branch below).
    drive_removable, drive_remote, drive_cdrom, drive_ramdisk = 2, 4, 5, 6
    devices: list[StorageDevice] = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:\\"
        if not os.path.exists(root):
            continue
        try:
            drive_type = int(kernel32.GetDriveTypeW(ctypes.c_wchar_p(root)))
        except Exception:  # noqa: BLE001
            drive_type = 0
        if drive_type in (0, 1, drive_cdrom):  # unknown / no root dir / CD-ROM — skip
            continue
        removable = drive_type == drive_removable
        if drive_type == drive_remote:
            interface = StorageInterface.network
        elif drive_type == drive_ramdisk:
            interface = StorageInterface.virtual
        elif removable:
            interface = StorageInterface.usb
        else:  # DRIVE_FIXED — SSD vs HDD isn't cheaply knowable without WMI/DeviceIoControl.
            interface = StorageInterface.unknown
        total, free = _disk_usage(root)
        devices.append(
            StorageDevice(
                mount_point=root,
                filesystem=_windows_filesystem(kernel32, root),
                interface=interface,
                total_bytes=total,
                free_bytes=free,
                removable=removable,
                rotational=None,
                cloud_synced=is_cloud_synced_path(root),
                device_name=f"{letter}:",
            )
        )
    return devices


def _windows_filesystem(kernel32: object, root: str) -> str:  # pragma: no cover - Windows-only; exercised on a real Windows host, not the Linux CI coverage runner.
    """Volume filesystem name (NTFS/FAT32/exFAT/...) via GetVolumeInformationW, or '' on failure."""
    try:
        import ctypes  # noqa: PLC0415

        fs_buf = ctypes.create_unicode_buffer(64)
        ok = kernel32.GetVolumeInformationW(  # type: ignore[attr-defined]
            ctypes.c_wchar_p(root), None, 0, None, None, None, fs_buf, ctypes.sizeof(fs_buf)
        )
        return fs_buf.value if ok else ""
    except Exception:  # noqa: BLE001
        return ""


def _detect_generic_devices() -> list[StorageDevice]:
    """Last-resort single-device fallback: characterize the root filesystem only."""
    root = "C:\\" if sys.platform == "win32" else "/"
    total, free = _disk_usage(root)
    if total is None and free is None:
        return []
    return [
        StorageDevice(
            mount_point=root,
            interface=StorageInterface.unknown,
            total_bytes=total,
            free_bytes=free,
            cloud_synced=False,
        )
    ]


def detect_storage_devices() -> list[StorageDevice]:
    """Characterize the host's mounted storage. Dispatches by platform; best-effort and never raises
    — an undetectable host yields ``[]`` rather than an exception."""
    try:
        if sys.platform == "win32":
            return _detect_windows_devices()
        if sys.platform.startswith("linux"):
            return _detect_linux_devices()
        return _detect_generic_devices()
    except Exception:  # noqa: BLE001 - detection must never crash the caller.
        return []


def find_device_for_path(path: str, devices: list[StorageDevice]) -> StorageDevice | None:
    """The device whose mount point is the longest prefix of ``path`` — i.e. the volume that path
    actually lives on. None when nothing matches (an undetected or unmounted path)."""
    try:
        target = os.path.abspath(path)
    except (OSError, ValueError):
        target = path
    best: StorageDevice | None = None
    best_len = -1
    for device in devices:
        mount = os.path.abspath(device.mount_point)
        # Normalize for a prefix compare (case-insensitive on Windows).
        norm_target = target.lower() if sys.platform == "win32" else target
        norm_mount = mount.lower() if sys.platform == "win32" else mount
        if norm_target == norm_mount or norm_target.startswith(norm_mount.rstrip("\\/") + os.sep):
            if len(norm_mount) > best_len:
                best, best_len = device, len(norm_mount)
    return best


def assess_path_for_role(
    path: str, role: StorageRole, devices: list[StorageDevice] | None = None
) -> StorageRoleAssessment:
    """Resolve the device for ``path`` and return the per-role suitability verdict. Detects the host
    storage when ``devices`` is not supplied (inject it in tests to stay off the real host)."""
    device_list = detect_storage_devices() if devices is None else devices
    device = find_device_for_path(path, device_list)
    return assess_role(
        role=role,
        path=path,
        device=device,
        inside_source_repo=is_inside_source_repo(path),
    )


def build_storage_profile(
    role_paths: dict[StorageRole, str] | None = None,
) -> StorageProfile:
    """Characterize the host's storage and, for any ``{role: path}`` requested, attach the per-role
    suitability assessment. Pure w.r.t. the project filesystem (reads mount/usage metadata only)."""
    devices = detect_storage_devices()
    assessments: list[StorageRoleAssessment] = []
    for role, path in (role_paths or {}).items():
        device = find_device_for_path(path, devices)
        assessments.append(
            assess_role(
                role=role,
                path=path,
                device=device,
                inside_source_repo=is_inside_source_repo(path),
            )
        )
    notes: list[str] = []
    if not devices:
        notes.append("no storage devices could be characterized on this host")
    return StorageProfile(
        captured_at=_now_iso(),
        devices=devices,
        assessments=assessments,
        notes=notes,
    )
