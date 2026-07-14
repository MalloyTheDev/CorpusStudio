# Running CorpusStudio training on native Linux (and reaching seq-4096)

This is the current verification runbook for long-sequence 7B QLoRA on a 12 GB card. The native-Linux
host is assembled, and its managed `backend-corpus-studio` environment passed the exact minimal
hardware-probe tuple. It is not a record of a completed native-Linux real workload or offload run.

> **Current evidence boundary:** native Windows/WDDM and WSL2 workload results remain labeled where
> measured. On native Linux, only the managed-environment CUDA-allocation, 4-bit-construction, minimal
> forward/backward, and math-SDPA probe is verified. Native-Linux real-workload training,
> bare-Linux FlashAttention for that workload, DeepSpeed/FSDP, CPU/NVMe offload fit, PCIe/NVMe
> throughput and sustained writes, full-sequence 7B success, and MoE runtime capability are unverified.

## Why native Linux (not WSL) for long sequences

CorpusStudio treats WSL and native Linux as distinct platforms. Windows and WSL have historical
workload measurements; native Linux currently has only the separate managed-environment probe:

| | native Windows | WSL2 | **native Linux** |
|---|---|---|---|
| flash attention (Blackwell) | deadlocks (measured WDDM) | works (measured) | math-SDPA env probe passes; real-workload flash unverified |
| over-VRAM behaviour | spills to shared RAM (measured) | spills, then wedges at scale (measured) | **unverified** |
| true seq-4096 7B QLoRA | impractical spill | fails (`device not ready`) | **unverified** |
| GPU access | WDDM | GPU-PV (paravirtualised) | direct PCIe; minimal managed-environment probe verified |

At true seq-4096 a 7B QLoRA is expected to exceed the 12 GB card. WSL2 measurements wedge when it
spills that hard. Native Linux is now the direct-GPU test bed for explicit offload, but its OOM
behavior and offload fit must be measured rather than inferred from the environment probe.

**Measured on a real RTX 5070 under WSL2** (true full-length sequences — QLoRA r16, grad checkpointing):

| seq | GPU peak | step time | verdict |
|----:|---------:|----------:|---------|
| 2048 | 14.1 GB | ~670 s | spills to RAM (impractically slow) |
| 2560 | 18.8 GB | ~310 s | spills |
| 3072 | 24.4 GB | ~460 s | spills — the **usable WSL ceiling**, barely |
| 3584 | — | — | **fails** (`device not ready`) |

So WSL "works" only up to seq-3072 and only by spilling at **5–11 minutes per step** — unusable for
real training. (Short *effective* sequences — the WBG corpus is ~1.2 k tokens — stay under 12 GB and
train fast at NATIVE_SAFE regardless of the `sequence_len` config.) True long-context needs the
native-Linux/offload experiment below, or more VRAM; the former is not yet proven at workload level.

## Historical NVMe preparation (completed 2026-07-13)

The following preparation commands are retained only as a rebuild reference. The current host and
`/mnt/training-nvme` layout already satisfy this step; do not rerun it as routine setup. On a replacement
host, update the OS and install host diagnostics/build prerequisites:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y \
  git curl wget rsync tmux htop jq \
  build-essential cmake ninja-build pkg-config \
  python3 python3-dev python3-venv \
  linux-headers-$(uname -r) \
  pciutils nvme-cli smartmontools sysstat iotop-c
```

Verify the install is portable and `/etc/fstab` uses UUIDs, never names such as `/dev/nvme0n1p2`:

```bash
lsblk -o NAME,MODEL,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINTS
cat /etc/fstab
findmnt /boot/efi
sudo update-initramfs -u -k all
sudo update-grub
```

For a replacement-host rebuild, prepare the training mount before installing NVIDIA/CUDA:

```bash
sudo mkdir -p \
  /mnt/training-nvme/environments \
  /mnt/training-nvme/hf-cache \
  /mnt/training-nvme/models \
  /mnt/training-nvme/datasets/raw \
  /mnt/training-nvme/datasets/prepared \
  /mnt/training-nvme/checkpoints \
  /mnt/training-nvme/runs \
  /mnt/training-nvme/artifacts \
  /mnt/training-nvme/evaluations \
  /mnt/training-nvme/offload/parameters \
  /mnt/training-nvme/offload/optimizer \
  /mnt/training-nvme/scratch \
  /mnt/training-nvme/tmp
sudo chown -R "$USER:$USER" /mnt/training-nvme
chmod 700 /mnt/training-nvme/offload
chmod 1777 /mnt/training-nvme/tmp
```

That sequencing is complete on the current RTX 5070 host. Do not repeat these preparation or install
steps unless performing a reviewed host rebuild.

## 1. Historical rebuild reference: dedicate an NVMe + install Ubuntu

Dedicate one NVMe to Linux (better than an external drive: full PCIe speed, native ext4 I/O). To
protect the Windows bootloader, **temporarily disconnect/disable the other drives in BIOS during the
Ubuntu install** so GRUB only writes to the Linux NVMe, then re-enable them and pick the boot drive
from the UEFI boot menu.

**Distro:** **Ubuntu 24.04 LTS** — genuinely the best for AI (best NVIDIA/CUDA support, the most-
tested target for every ML library and Docker image). If the Blackwell driver step worries you,
**Pop!_OS 22.04** (Ubuntu-based) ships the NVIDIA drivers pre-configured and the `apt`-based bootstrap
here works on it unchanged. Skip Debian (packages too old for Blackwell) and Arch/NixOS (high-
maintenance for a stable training box).

**Headless Server vs Desktop → go headless** on a tight-VRAM card. A desktop compositor holds
~300–500 MB of VRAM just to draw the screen — VRAM you need to reach seq-4096. **Ubuntu Server**
frees it, uses less RAM/CPU, and is more stable. You lose nothing for training: the CorpusStudio
*engine* (train-check / platform-plan / platform-run) is CLI and runs fully headless. **SSH in from
Windows** — or use **VS Code Remote-SSH** (editor + terminal + file browser over SSH, feels local).
The clean split: **author + prep datasets on the Windows CorpusStudio app, run the heavy training on
the headless Linux box** — exactly the platform's headless-engine + swappable-shell design. Only pick
Ubuntu Desktop if you specifically want a GUI on the Linux box (accept the VRAM cost).

## 2. NVIDIA driver (the one gotcha for Blackwell / sm_120)

The current host already satisfies this prerequisite and its verified driver/GPU facts are recorded in
[`HOST_STATE.md`](HOST_STATE.md). Run this section only during a reviewed replacement-host rebuild,
after the NVMe PCIe link, mount, and health have been inspected.

The RTX 50-series needs a recent driver (**570+**) and a recent kernel:

```bash
sudo ubuntu-drivers install         # or: sudo add-apt-repository ppa:graphics-drivers/ppa
sudo reboot
nvidia-smi                          # must show the RTX 5070 before continuing
```

If Secure Boot is on, the installer prompts you to **enrol a MOK key** — do it (a reboot + a blue
MOK-manager screen). CUDA `cu128` (which supports sm_120) is pulled in by the Python step below, not
a system CUDA toolkit.

## 3. Get CorpusStudio + your data onto Linux

The active checkout is already on the Linux training filesystem:

```bash
cd /mnt/training-nvme/repos/CorpusStudio
git status --short --branch
```

Clone from the reviewed upstream only when rebuilding a new host. Never use the old `C:` or `F:`
checkouts under `/mnt/windows-c` or `/mnt/windows-f` as active development roots.

Two things are **not** in the repo:

- **Your datasets** (e.g. the World Bible Generator JSONL splits). Historical source material remains
  readable under `/mnt/windows-c` and `/mnt/windows-f`, but those mounts are history-only project
  inputs. Copy approved mutable training inputs to `/mnt/training-nvme/datasets/` and never develop
  from or write through the old Windows repository checkouts.
- **The base model weights** (~15 GB Qwen). The current host sets `HF_HOME` to
  `/mnt/training-nvme/cache/huggingface`; inspect snapshot completeness and license evidence before
  any `model-fetch`. Keep model downloads on `/mnt/training-nvme`, not the root filesystem or Windows
  mounts.

## 4. Verify the existing managed training environment

```bash
cd /mnt/training-nvme/repos/CorpusStudio
engine/.venv/bin/corpus-studio env-status backend-corpus-studio --refresh
engine/.venv/bin/corpus-studio env-probe backend-corpus-studio
```

On this host, the managed environment lives beneath the XDG data root documented in
[`HOST_STATE.md`](HOST_STATE.md) and is `HARDWARE_VERIFIED` for its exact minimal probe tuple. Use
`env-plan` and `env-create` only for an explicitly reviewed recreation; creation performs network
package installation. Package installation or an environment probe alone is not workload support.
The `scripts/setup_linux_training.sh` path is a manual diagnostic fallback, not the managed or offload
backend workflow.

## 5. Reaching seq-4096 — the honest playbook

The current estimates say seq-4096 7B QLoRA will not fit in 12 GB unaided. Candidate experiments,
cheapest first, are:

1. **Fused-CE loss** — first add and pass a complete tuple probe for the Liger + selected optimizer
   combination, then measure its effect on this host. A standalone Liger field/package result is not
   execution support.
2. **Activation offload to CPU RAM** — a planned long-sequence lever that needs a real isolated backend
   and measurement on the current native-Linux host.
3. **DeepSpeed ZeRO offload → NVMe** — planned only. CorpusStudio does not yet ship or verify this
   backend, and NVMe offload must not be inferred from the physical `RunPlan` contract.
4. **Multiple GPUs** — a future FSDP/DeepSpeed path, not a verified current capability. Heterogeneous
   GPU behavior, usable combined memory, communication cost, and per-device limits all require a real
   backend and measurements.
5. **A bigger single card** — the clean answer; removes the whole problem.

The existing singleton baseline can be planned and supervised with the platform, but this command
does not implement or prove CPU/NVMe offload:

```bash
cd /mnt/training-nvme/repos/CorpusStudio
engine/.venv/bin/corpus-studio platform-plan --base-model Qwen/Qwen2.5-7B-Instruct \
    --model-revision a09a35458c702b33eeacc393d103063234e8bc28 --dataset train.jsonl \
    --dataset-format chat --chat-template-sha256 "$CHAT_TEMPLATE_SHA256" \
    --sequence-len 1024 --out /tmp/plan
engine/.venv/bin/corpus-studio platform-run /tmp/plan/RunPlan.json --subprocess --out ./run
```

For a chat dataset, also supply the exact `--chat-template-sha256`; omission is a blocking preflight
error rather than a formatting fallback.

**Expectations, honestly:** no completion or fit claim exists yet. Establish the sequence-1024 native
baseline first, increase sequence length gradually, then test CPU offload, and attempt NVMe offload
only after baseline GPU and non-destructive storage measurements exist.

> **Status:** historical WSL2 workload measurements and the current native-Linux managed-environment
> probe are verified only within their separately labeled boundaries. Native-Linux real-workload and
> offload paths remain unverified; continue the ordered baseline-to-offload measurements above.
