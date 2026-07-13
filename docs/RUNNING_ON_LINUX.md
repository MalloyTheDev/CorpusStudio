# Running CorpusStudio training on native Linux (and reaching seq-4096)

This is the path to **true long-sequence 7B QLoRA** on a 12 GB card — the thing WSL2 can't do.

## Why native Linux (not WSL) for long sequences

CorpusStudio treats WSL and Linux as distinct platforms for good reason (verified on an RTX 5070 this
project):

| | native Windows | WSL2 | **native Linux** |
|---|---|---|---|
| flash attention (Blackwell) | ✗ deadlocks (WDDM) | ✓ works | ✓ works |
| over-VRAM behaviour | spills to shared RAM (slow) | spills, then **wedges** at scale | **hard-OOM** (clean) |
| true seq-4096 7B QLoRA | math-attn spill | **fails** (`device not ready` — GPU-PV wall) | possible **with offload** |
| GPU access | WDDM | GPU-PV (paravirtualised) | **direct** |

At true seq-4096 a 7B QLoRA needs ~15–20 GB — over the 12 GB card. On WSL2 the GPU-PV layer wedges
when it spills that hard; **native Linux gives direct GPU access + clean OOM**, so you can add
*explicit* offload (CPU/NVMe) to fit the overflow instead of relying on a fragile silent spill.

## 1. Dedicate an NVMe + install Ubuntu

You have 3 NVMes — dedicate one to Linux (better than an external drive: full PCIe speed, native
ext4 I/O). To protect the Windows bootloader, **temporarily disconnect/disable the other drives in
BIOS during the Ubuntu install** so GRUB only writes to the Linux NVMe, then re-enable them and pick
the boot drive from the UEFI boot menu.

- **Ubuntu 24.04 LTS** (best NVIDIA/CUDA support).

## 2. NVIDIA driver (the one gotcha for Blackwell / sm_120)

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

## 4. Build the training env (no sudo)

```bash
bash scripts/setup_linux_training.sh        # uv + Python 3.12 + torch cu128 + the stack + engine
source ~/cs-train/bin/activate
corpus-studio train-check                   # expect: READY (GPU QLoRA), sees the 5070
```

This mirrors the exact stack verified on WSL (torch 2.11.0+cu128, transformers 5.13.1, trl 1.8.0,
peft 0.19.0, bnb 0.49.2) **plus** `deepspeed` + `liger-kernel` for the offload/long-seq levers.

## 5. Reaching seq-4096 — the honest playbook

seq-4096 7B QLoRA does **not** fit in 12 GB unaided on any platform. On native Linux the options,
cheapest first:

1. **Fused-CE loss** (removes the ~2.5 GB fp32 logits spike) — CorpusStudio's `--memory-efficient` /
   `--use-liger`. Necessary, not sufficient on its own.
2. **Activation offload to CPU RAM** — the real long-seq lever (activations dominate). On native Linux
   the pinned-memory transfer path is robust (it is the WSL2 GPU-PV layer, not Linux, that made this
   fail here). This is what lets the overflow live in your system RAM.
3. **DeepSpeed ZeRO offload → NVMe** — for when RAM is also tight; point it at one of your NVMes. Best
   for full fine-tuning; for QLoRA the offloadable optimizer is small, so prefer (1)+(2) first.
4. **More VRAM** — the clean answer; a bigger card removes the whole problem.

Run it through the platform (the planner seals the levers, the watchdog measures the real fit, and on
Linux a genuine over-VRAM config now OOMs cleanly with an actionable message instead of wedging):

```bash
corpus-studio platform-plan --base-model Qwen/Qwen2.5-7B-Instruct --dataset train.jsonl \
    --dataset-format chat --sequence-len 4096 --memory-efficient --out /tmp/plan
corpus-studio platform-run /tmp/plan/RunPlan.json --runner training --subprocess --out ./run
```

**Expectations, honestly:** with offload, expect long-seq steps to be **slow** (PCIe transfers) but to
**complete** — the point of native Linux is that a too-big config degrades to slow-but-training or a
clean OOM, not the confusing WSL2 wedge. If your real data is short (the WBG corpus is ~1.2 k tokens),
you do **not** need true seq-4096 — the `sequence_len=4096` *config* already trains fine at
NATIVE_SAFE because the effective sequences are short.

> **Status:** this stack is verified on WSL2; the native-Linux path + the offload-to-fit-seq-4096
> claim are **unverified until booted on real Linux**. Boot Ubuntu on the NVMe, run the bootstrap, and
> we can measure the actual seq-4096 fit together.
