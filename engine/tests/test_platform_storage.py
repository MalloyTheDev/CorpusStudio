"""Storage topology detection + the per-role safe-spill suitability guardrail.

The suitability logic (:func:`assess_role`) is PURE over an injected StorageDevice, so every verdict is
proven here without touching the real host; the detection functions get a smoke test that they run
cross-platform without crashing.
"""

from __future__ import annotations

from corpus_studio.platform.contracts import StorageDevice, StorageProfile
from corpus_studio.platform.enums import StorageInterface, StorageRole, StorageSuitability
from corpus_studio.platform.storage_profiler import (
    assess_path_for_role,
    assess_role,
    build_storage_profile,
    classify_interface,
    detect_storage_devices,
    find_device_for_path,
    is_cloud_synced_path,
    min_free_bytes_for_role,
    parse_proc_mounts,
)

_GB = 1_000_000_000


def _device(**overrides) -> StorageDevice:
    base = dict(
        mount_point="/data",
        filesystem="ext4",
        interface=StorageInterface.nvme_pcie,
        total_bytes=1000 * _GB,
        free_bytes=500 * _GB,
        removable=False,
        rotational=False,
        cloud_synced=False,
    )
    base.update(overrides)
    return StorageDevice(**base)


# ---- pure classifiers --------------------------------------------------------


def test_classify_interface_prefers_removable_then_nvme_then_rotational():
    assert classify_interface("/dev/nvme0n1", removable=True, rotational=False) == StorageInterface.usb
    assert classify_interface("/dev/nvme0n1", removable=False, rotational=False) == StorageInterface.nvme_pcie
    assert classify_interface("/dev/sda", removable=False, rotational=True) == StorageInterface.hdd
    assert classify_interface("/dev/sda", removable=False, rotational=False) == StorageInterface.sata_ssd
    assert classify_interface(None, removable=None, rotational=None) == StorageInterface.unknown


def test_is_cloud_synced_path_matches_known_clients_only():
    assert is_cloud_synced_path("/home/me/Dropbox/project")
    assert is_cloud_synced_path("C:\\Users\\me\\OneDrive - Acme\\ck")
    assert is_cloud_synced_path("/Users/me/Google Drive/x")
    # A bare "sync"/"box" folder is NOT flagged (too generic → false positives).
    assert not is_cloud_synced_path("/home/me/sync/data")
    assert not is_cloud_synced_path("/data/checkpoints")


def test_parse_proc_mounts_filters_pseudo_and_unescapes_spaces():
    text = (
        "proc /proc proc rw 0 0\n"
        "sysfs /sys sysfs rw 0 0\n"
        "/dev/nvme0n1p2 / ext4 rw 0 0\n"
        "/dev/sdb1 /mnt/my\\040disk ext4 rw 0 0\n"
        "tmpfs /run tmpfs rw 0 0\n"
    )
    rows = parse_proc_mounts(text)
    # proc + sysfs dropped; the real ext4 mounts + tmpfs (a real fs, classified virtual later) kept.
    assert ("/dev/nvme0n1p2", "/", "ext4") in rows
    assert ("/dev/sdb1", "/mnt/my disk", "ext4") in rows  # \040 → space
    assert not any(fs in {"proc", "sysfs"} for _, _, fs in rows)


def test_min_free_bytes_for_role():
    assert min_free_bytes_for_role(StorageRole.optimizer_offload) == 20 * _GB
    assert min_free_bytes_for_role(StorageRole.checkpoints) == 20 * _GB


# ---- the suitability guardrail (pure) ----------------------------------------


def test_assess_none_device_is_unknown_never_a_false_suitable():
    a = assess_role(role=StorageRole.checkpoints, path="/x", device=None, inside_source_repo=False)
    assert a.suitability == StorageSuitability.unknown
    assert a.reasons


def test_nvme_with_headroom_is_suitable_for_offload():
    a = assess_role(
        role=StorageRole.optimizer_offload,
        path="/data/offload",
        device=_device(interface=StorageInterface.nvme_pcie, free_bytes=200 * _GB),
        inside_source_repo=False,
    )
    assert a.suitability == StorageSuitability.suitable and not a.reasons


def test_usb_is_unsuitable_for_offload():
    a = assess_role(
        role=StorageRole.optimizer_offload,
        path="/media/usb/offload",
        device=_device(interface=StorageInterface.usb, removable=True, free_bytes=200 * _GB),
        inside_source_repo=False,
    )
    assert a.suitability == StorageSuitability.unsuitable
    assert any("USB" in r for r in a.reasons)


def test_usb_is_fine_for_archive_role_specific():
    # The SAME USB device is suitable for archival (low write) — suitability is judged per role.
    a = assess_role(
        role=StorageRole.archive,
        path="/media/usb/archive",
        device=_device(interface=StorageInterface.usb, removable=True, free_bytes=200 * _GB),
        inside_source_repo=False,
    )
    assert a.suitability == StorageSuitability.suitable


def test_network_mount_is_unsuitable_for_checkpoints():
    a = assess_role(
        role=StorageRole.checkpoints,
        path="/net/share/ck",
        device=_device(interface=StorageInterface.network, free_bytes=500 * _GB),
        inside_source_repo=False,
    )
    assert a.suitability == StorageSuitability.unsuitable
    assert any("network" in r for r in a.reasons)


def test_cloud_synced_folder_is_unsuitable_for_high_write():
    a = assess_role(
        role=StorageRole.checkpoints,
        path="/home/me/Dropbox/ck",
        device=_device(cloud_synced=True, free_bytes=500 * _GB),
        inside_source_repo=False,
    )
    assert a.suitability == StorageSuitability.unsuitable
    assert any("cloud-sync" in r for r in a.reasons)


def test_inside_source_repo_is_unsuitable_for_scratch():
    a = assess_role(
        role=StorageRole.scratch,
        path="/repo/.scratch",
        device=_device(free_bytes=500 * _GB),
        inside_source_repo=True,
    )
    assert a.suitability == StorageSuitability.unsuitable
    assert any("source repository" in r for r in a.reasons)


def test_rotational_disk_is_marginal_for_offload_not_forbidden():
    a = assess_role(
        role=StorageRole.parameter_offload,
        path="/data/offload",
        device=_device(interface=StorageInterface.hdd, rotational=True, free_bytes=500 * _GB),
        inside_source_repo=False,
    )
    assert a.suitability == StorageSuitability.marginal
    assert any("rotational" in r for r in a.reasons)


def test_nearly_full_disk_is_unsuitable():
    a = assess_role(
        role=StorageRole.optimizer_offload,
        path="/data/offload",
        device=_device(interface=StorageInterface.nvme_pcie, free_bytes=2 * _GB),  # needs ~20 GB
        inside_source_repo=False,
    )
    assert a.suitability == StorageSuitability.unsuitable
    assert any("free" in r for r in a.reasons)


def test_unsuitable_reason_wins_over_marginal():
    # A rotational (marginal) disk that is ALSO nearly full (unsuitable) → unsuitable overall.
    a = assess_role(
        role=StorageRole.optimizer_offload,
        path="/data/offload",
        device=_device(interface=StorageInterface.hdd, rotational=True, free_bytes=1 * _GB),
        inside_source_repo=False,
    )
    assert a.suitability == StorageSuitability.unsuitable


# ---- device resolution + wiring ----------------------------------------------


def test_find_device_for_path_picks_the_longest_matching_mount(tmp_path):
    parent = _device(mount_point=str(tmp_path))
    child_mount = tmp_path / "sub"
    child_mount.mkdir()
    child = _device(mount_point=str(child_mount), interface=StorageInterface.sata_ssd)
    target = child_mount / "offload"
    found = find_device_for_path(str(target), [parent, child])
    assert found is not None and found.mount_point == str(child_mount)


def test_assess_path_for_role_with_injected_devices(tmp_path):
    # model_cache is not repo-sensitive, so the verdict depends only on the device — deterministic
    # regardless of whether tmp_path happens to sit inside a git tree.
    device = _device(mount_point=str(tmp_path), interface=StorageInterface.nvme_pcie, free_bytes=200 * _GB)
    a = assess_path_for_role(str(tmp_path / "models"), StorageRole.model_cache, devices=[device])
    assert a.role == StorageRole.model_cache
    assert a.suitability == StorageSuitability.suitable


# ---- detection smoke + contract round-trip -----------------------------------


def test_detect_storage_devices_runs_without_crashing():
    devices = detect_storage_devices()
    assert isinstance(devices, list)
    for device in devices:
        assert isinstance(device, StorageDevice) and device.mount_point


def test_build_storage_profile_smoke_and_round_trip():
    profile = build_storage_profile()
    assert profile.captured_at
    restored = StorageProfile.model_validate_json(profile.model_dump_json())
    assert restored.contract_version == profile.contract_version
    assert len(restored.devices) == len(profile.devices)
