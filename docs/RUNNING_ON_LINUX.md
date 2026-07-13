# Running CorpusStudio training on native Linux (and reaching seq-4096)

This is the planned verification runbook for long-sequence 7B QLoRA on a 12 GB card. It is not a
record of a completed native-Linux or offload run.

> **Current evidence boundary:** native Windows/WDDM and WSL2 results are labeled where measured.
> Native-Linux RTX 5070 training, bare-Linux FlashAttention, DeepSpeed/FSDP, CPU/NVMe offload fit,
> PCIe/NVMe throughput and sustained writes, full-sequence 7B success, and MoE runtime capability are
> all unverified until the Linux NVMe is installed in the final desktop and those tests are run.

## Why native Linux (not WSL) for long sequences

CorpusStudio treats WSL and native Linux as distinct platforms. Only the Windows and WSL columns have
project measurements today:

| | native Windows | WSL2 | **native Linux** |
|---|---|---|---|
| flash attention (Blackwell) | deadlocks (measured WDDM) | works (measured) | **unverified** |
| over-VRAM behaviour | spills to shared RAM (measured) | spills, then wedges at scale (measured) | **unverified** |
| true seq-4096 7B QLoRA | impractical spill | fails (`device not ready`) | **unverified** |
| GPU access | WDDM | GPU-PV (paravirtualised) | direct access expected; verify after install |

At true seq-4096 a 7B QLoRA is expected to exceed the 12 GB card. WSL2 measurements wedge when it
spills that hard. Native Linux is the planned direct-GPU test bed for explicit offload, but its OOM
behavior and offload fit must be measured rather than inferred.

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
native-Linux/offload experiment below, or more VRAM; the former is not yet proven.

## Before the adapter arrives: portable NVMe preparation only

On the temporary Linux computer, update the OS and install host diagnostics/build prerequisites:

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

Prepare the future training mount without installing NVIDIA/CUDA yet:

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

Do not install the NVIDIA driver, CUDA userspace stack, or managed training environment until this
NVMe is installed in the RTX 5070 desktop through the PCIe adapter.

## 1. Dedicate an NVMe + install Ubuntu

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

Run this section only after the prepared NVMe is installed in the RTX 5070 desktop and its PCIe link,
mount, and health have been inspected.

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

**The code: `git clone` it — no zip.** CorpusStudio lives on GitHub, so pull it straight down (and
`git pull` later for updates, which a zip can't do):

```bash
sudo apt install -y git                                   # + `gh` if the repo is private
git clone https://github.com/MalloyTheDev/CorpusStudio.git
cd CorpusStudio
```

For a private repo, authenticate first: `sudo apt install -y gh && gh auth login`, then clone.

Two things are **not** in the repo:

- **Your datasets** (e.g. the World Bible Generator JSONL splits). If you **dual-boot** on the same
  machine, the cleanest route is to **mount the Windows NVMe from Linux and read them directly** — no
  copy:
  ```bash
  lsblk -f                                  # find the Windows NTFS partition, e.g. /dev/nvme0n1p3
  sudo mkdir -p /mnt/win && sudo mount -t ntfs3 /dev/nvme0n1p3 /mnt/win
  # datasets now at /mnt/win/WorldBibleGenerator/…  — point --dataset straight at them
  ```
  Otherwise copy the dataset folder over (USB / the mounted drive).
- **The base model weights** (~15 GB Qwen). They **re-download automatically** the first time
  (`corpus-studio model-fetch …`, or on the first run). To skip the download, copy the HF cache:
  `C:\Users\<you>\.cache\huggingface` → `~/.cache/huggingface`.

## 4. Build the managed training environment (no sudo)

```bash
python3 -m venv /mnt/training-nvme/environments/control-plane
/mnt/training-nvme/environments/control-plane/bin/pip install -e ./engine
source /mnt/training-nvme/environments/control-plane/bin/activate

corpus-studio env-runtimes --recipe backend-corpus-studio
corpus-studio env-plan backend-corpus-studio \
  --env-id backend-corpus-studio \
  --runtime /usr/bin/python3 \
  --accelerator cu128 \
  --manager-root /mnt/training-nvme/environments/manager
# Review the exact argv, indexes, target, and size, then repeat with:
corpus-studio env-create backend-corpus-studio \
  --env-id backend-corpus-studio \
  --runtime /usr/bin/python3 \
  --accelerator cu128 \
  --manager-root /mnt/training-nvme/environments/manager \
  --confirm <resolution-hash>
corpus-studio env-probe backend-corpus-studio \
  --manager-root /mnt/training-nvme/environments/manager --json
```

This step happens only on the final machine after `nvidia-smi` succeeds. The managed environment must
be built and probed there; package installation alone is not backend or hardware support. The
`scripts/setup_linux_training.sh` path is a manual diagnostic fallback, not the managed or offload
backend workflow.

## 5. Reaching seq-4096 — the honest playbook

The current estimates say seq-4096 7B QLoRA will not fit in 12 GB unaided. Candidate experiments,
cheapest first, are:

1. **Fused-CE loss** — CorpusStudio's `--memory-efficient` / `--use-liger`; measure its effect on this
   host rather than carrying forward the Windows/WSL estimate.
2. **Activation offload to CPU RAM** — a planned long-sequence lever that needs a real isolated backend
   and measurement on the final host.
3. **DeepSpeed ZeRO offload → NVMe** — planned only. CorpusStudio does not yet ship or verify this
   backend, and NVMe offload must not be inferred from the physical `RunPlan` contract.
4. **Multiple GPUs** — a future FSDP/DeepSpeed path, not a verified current capability. Heterogeneous
   GPU behavior, usable combined memory, communication cost, and per-device limits all require a real
   backend and measurements.
5. **A bigger single card** — the clean answer; removes the whole problem.

The existing singleton baseline can be planned and supervised with the platform, but this command
does not implement or prove CPU/NVMe offload:

```bash
corpus-studio platform-plan --base-model Qwen/Qwen2.5-7B-Instruct --dataset train.jsonl \
    --dataset-format chat --sequence-len 1024 --memory-efficient --out /tmp/plan
corpus-studio platform-run /tmp/plan/RunPlan.json --runner training --subprocess --out ./run
```

**Expectations, honestly:** no completion or fit claim exists yet. Establish the sequence-1024 native
baseline first, increase sequence length gradually, then test CPU offload, and attempt NVMe offload
only after baseline GPU and non-destructive storage measurements exist.

> **Status:** this stack is verified only where explicitly labeled on WSL2; the native-Linux and
> offload paths remain unverified. After the adapter/final-machine installation, follow the ordered
> baseline-to-offload measurements above.
