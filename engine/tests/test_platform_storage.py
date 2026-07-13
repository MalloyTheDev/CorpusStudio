"""Storage topology detection + the per-role safe-spill suitability guardrail.

The suitability logic (:func:`assess_role`) is PURE over an injected StorageDevice, so every verdict is
proven here without touching the real host; the detection functions get a smoke test that they run
cross-platform without crashing.
"""

from __future__ import annotations

from corpus_studio.platform.contracts import StorageDevice, StorageProfile
from corpus_studio.platform.enums import StorageInterface, StorageRole, StorageSuitability
from corpus_studio.platform.storage_profiler import (
    _detect_generic_devices,
    _linux_block_flag,
    assess_path_for_role,
    assess_role,
    build_storage_profile,
    classify_interface,
    classify_linux_mount,
    classify_storage_failure,
    detect_storage_devices,
    find_device_for_path,
    is_cloud_synced_path,
    is_inside_source_repo,
    min_free_bytes_for_role,
    parse_proc_mounts,
    recommended_role_placement,
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
        wsl_host_drive=False,
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
        "\n"  # blank / malformed line (< 3 fields) is skipped, not a crash
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


def test_usb_is_marginal_for_load_latency_roles():
    # A USB SSD is not high-write for the model cache, but its latency drags shard/dataset loading.
    a = assess_role(
        role=StorageRole.model_cache,
        path="/media/usb/models",
        device=_device(interface=StorageInterface.usb, removable=True, free_bytes=500 * _GB),
        inside_source_repo=False,
    )
    assert a.suitability == StorageSuitability.marginal
    assert any("latency" in r for r in a.reasons)


def test_usb_is_marginal_for_small_file_roles():
    a = assess_role(
        role=StorageRole.python_env,
        path="/media/usb/.venv",
        device=_device(interface=StorageInterface.usb, removable=True, free_bytes=500 * _GB),
        inside_source_repo=False,
    )
    assert a.suitability == StorageSuitability.marginal
    assert any("small files" in r for r in a.reasons)


def test_wsl_host_drive_is_unsuitable_for_the_venv():
    # The venv/repo on /mnt/f (a Windows drive from WSL) is the worst case — NTFS translation makes
    # thousands of small-file imports crawl.
    a = assess_role(
        role=StorageRole.python_env,
        path="/mnt/f/CorpusStudio/.venv",
        device=_device(interface=StorageInterface.virtual, wsl_host_drive=True, free_bytes=500 * _GB),
        inside_source_repo=False,
    )
    assert a.suitability == StorageSuitability.unsuitable
    assert any("WSL" in r for r in a.reasons)


def test_wsl_host_drive_is_marginal_for_other_runtime_roles():
    a = assess_role(
        role=StorageRole.checkpoints,
        path="/mnt/f/ck",
        device=_device(interface=StorageInterface.virtual, wsl_host_drive=True, free_bytes=500 * _GB),
        inside_source_repo=False,
    )
    assert a.suitability == StorageSuitability.marginal
    assert any("WSL" in r for r in a.reasons)


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


def test_classify_linux_mount_wsl_host_drive_is_virtual_with_note():
    interface, removable, rotational, notes, wsl_host = classify_linux_mount(
        "D:", "/mnt/d", "drvfs", wsl=True
    )
    assert interface == StorageInterface.virtual
    assert removable is None and rotational is None
    assert notes and "WSL" in notes[0]
    assert wsl_host is True  # flagged so the role assessment can penalize small-file roles here


def test_classify_linux_mount_network_and_virtual_filesystems():
    net, *_ = classify_linux_mount("//srv/share", "/net", "nfs4", wsl=False)
    assert net == StorageInterface.network
    virt, *_ = classify_linux_mount("tmpfs", "/run", "tmpfs", wsl=False)
    assert virt == StorageInterface.virtual


def test_classify_linux_mount_real_disk_reads_block_attributes():
    # A device absent from /sys/block → attributes unknown; an nvme* name still classifies by name.
    interface, removable, rotational, notes, wsl_host = classify_linux_mount(
        "/dev/nvme9n1p1", "/data", "ext4", wsl=False
    )
    assert interface == StorageInterface.nvme_pcie
    assert removable is None and rotational is None and notes == []
    assert wsl_host is False


def test_detect_generic_devices_returns_the_root_volume():
    devices = _detect_generic_devices()
    assert isinstance(devices, list)
    for device in devices:
        assert isinstance(device, StorageDevice) and device.mount_point


def test_find_device_for_path_returns_none_when_nothing_matches():
    assert find_device_for_path("/nowhere/x", []) is None


def test_is_inside_source_repo_detects_a_git_tree(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "sub").mkdir()
    # A path under a .git tree is detected. (The False branch is exercised across the suite by the
    # many tmp paths that sit outside any git tree — its result depends on where tmp_path lives.)
    assert is_inside_source_repo(str(repo / "sub")) is True


def test_linux_block_flag_returns_none_for_absent_or_non_block_device():
    # A non-block device name short-circuits to None; absent block devices (nvme partition-suffix and
    # sd* digit-suffix parent-disk resolution) can't be read → None. Exercises both suffix branches.
    assert _linux_block_flag("/dev/loop0", "removable") is None
    assert _linux_block_flag("/dev/nvme9n1p1", "queue/rotational") is None
    assert _linux_block_flag("/dev/sdz9", "removable") is None


def test_disk_usage_error_path_returns_unknown():
    from corpus_studio.platform.storage_profiler import _disk_usage

    total, free = _disk_usage("/this/path/does/not/exist/anywhere-xyz")
    assert total is None and free is None


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


def test_build_storage_profile_attaches_role_assessments(tmp_path):
    profile = build_storage_profile({StorageRole.optimizer_offload: str(tmp_path / "offload")})
    assert len(profile.assessments) == 1
    assert profile.assessments[0].role == StorageRole.optimizer_offload


# ---- failure diagnostic + recommended placement ------------------------------


def test_classify_storage_failure_flags_io_errors():
    verdict, signals = classify_storage_failure(
        "OSError: [Errno 5] Input/output error while writing checkpoint-50"
    )
    assert verdict == "storage_implicated" and signals


def test_classify_storage_failure_exonerates_vram_and_kernel_failures():
    assert classify_storage_failure("torch.cuda.OutOfMemoryError: CUDA out of memory")[0] == "not_storage"
    assert classify_storage_failure("first backward hung on sm_120 flashattention")[0] == "not_storage"


def test_classify_storage_failure_ambiguous_or_empty_is_unknown():
    # Both kinds of signal → don't guess.
    assert classify_storage_failure("cuda error and Input/output error")[0] == "unknown"
    assert classify_storage_failure("training diverged")[0] == "unknown"


def test_recommended_role_placement_puts_archive_external_and_training_on_nvme():
    placement = recommended_role_placement()
    assert "USB" in placement[StorageRole.archive]
    assert "NVMe" in placement[StorageRole.checkpoints]
    assert "NVMe" in placement[StorageRole.model_cache]
    assert "SATA" in placement[StorageRole.python_env]  # the venv wants a reliable internal SSD
    # Every role has a recommendation.
    assert set(placement) == set(StorageRole)
