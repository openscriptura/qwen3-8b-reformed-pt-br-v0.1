"""
vastai_run_experiments.py — Automate Phase 2 experiments on vast.ai.

Workflow:
  1. Search for cheapest RTX 4090 (≥24GB VRAM, ≥40GB disk, CUDA 12.4+).
  2. Launch instance with PyTorch image + onstart script that:
       - Clones the OpenScriptura repo
       - Installs requirements
       - Runs dry-run to validate
       - Starts the requested experiment config
  3. Print SSH command + instance ID to monitor progress.
  4. (Optional) --destroy after training completes.

Usage:
  # Install vast CLI first:
  pip install vastai

  # Set your API key (one-time):
  vastai set api-key <YOUR_KEY>

  # Find available 4090s and show prices:
  python scripts/vastai_run_experiments.py --search

  # Launch experiment D on cheapest available 4090:
  python scripts/vastai_run_experiments.py --config configs/exp_d.yaml

  # Launch on a specific offer ID (from --search output):
  python scripts/vastai_run_experiments.py --config configs/exp_d.yaml --offer-id 12345678

  # Launch all 4 experiments sequentially on the same instance:
  python scripts/vastai_run_experiments.py --config configs/exp_d.yaml --all-configs

  # After training: check status
  python scripts/vastai_run_experiments.py --status <INSTANCE_ID>

  # Destroy instance when done:
  python scripts/vastai_run_experiments.py --destroy <INSTANCE_ID>

Requirements:
  pip install vastai
  vastai set api-key <YOUR_KEY>
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_REPO    = "https://github.com/openscriptura/qwen3-8b-reformed-pt-br-v0.1.git"
DOCKER_IMAGE   = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB        = 60          # Qwen3-8B ~16GB + deps ~5GB + checkpoints ~20GB + buffer
MIN_VRAM_GB    = 24
MIN_DISK_GB    = 50

# Search query: RTX 4090, verified, direct SSH, CUDA 12+
SEARCH_QUERY = (
    "gpu_name=RTX_4090 "
    "num_gpus=1 "
    "verified=True "
    "rentable=True "
    "cuda_vers>=12.4 "
    "disk_space>={min_disk} "
    "dph_total<2.0"          # under $2/hr (4090 should be $0.35–0.55)
)

ALL_CONFIGS = [
    "configs/exp_d.yaml",   # recommended first
    "configs/exp_c.yaml",
    "configs/exp_b.yaml",
    "configs/exp_a.yaml",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], capture: bool = True) -> tuple[int, str]:
    """Run a shell command, return (returncode, stdout+stderr)."""
    result = subprocess.run(
        cmd, capture_output=capture, text=True, encoding="utf-8", errors="replace"
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def _vastai(*args) -> tuple[int, str]:
    return _run(["vastai"] + list(args))


def _check_vastai() -> None:
    rc, out = _vastai("show", "user")
    if rc != 0:
        print("ERROR: vastai CLI not found or not authenticated.")
        print("  Install : pip install vastai")
        print("  Auth    : vastai set api-key <YOUR_KEY>")
        print("  Get key : https://cloud.vast.ai/account/")
        sys.exit(1)
    print("✓ vastai authenticated")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_offers(top_n: int = 10) -> list[dict]:
    """Return top_n cheapest RTX 4090 offers, sorted by dph_total."""
    query = SEARCH_QUERY.format(min_disk=MIN_DISK_GB)
    rc, out = _vastai("search", "offers", "--raw", query)
    if rc != 0:
        print(f"ERROR searching offers:\n{out}")
        sys.exit(1)
    try:
        offers = json.loads(out)
    except json.JSONDecodeError:
        print(f"ERROR parsing search results:\n{out[:500]}")
        sys.exit(1)

    # Sort by price ascending
    offers = sorted(offers, key=lambda o: o.get("dph_total", 999))
    return offers[:top_n]


def print_offers(offers: list[dict]) -> None:
    print(f"\n{'ID':>10}  {'GPU':20}  {'VRAM':>6}  {'$/hr':>6}  {'Disk':>6}  {'CUDA':>6}  {'Location'}")
    print("-" * 80)
    for o in offers:
        print(
            f"{o['id']:>10}  "
            f"{o.get('gpu_name','?'):20}  "
            f"{o.get('gpu_ram',0):>5.0f}G  "
            f"${o.get('dph_total',0):>5.3f}  "
            f"{o.get('disk_space',0):>5.0f}G  "
            f"{o.get('cuda_max_good','?'):>6}  "
            f"{o.get('geolocation','?')}"
        )
    print()


# ---------------------------------------------------------------------------
# Onstart script
# ---------------------------------------------------------------------------

def _build_onstart(config_path: str, all_configs: bool) -> str:
    """Build the bash onstart script that runs inside the vast.ai container."""

    if all_configs:
        # Run all 4 configs sequentially, log each to a separate file
        per_cfg_cmds = [
            f'echo "=== Starting {cfg} ===" | tee -a /workspace/training.log\n'
            f'python scripts/04_experiment.py --config {cfg} '
            f'>> /workspace/training_{Path(cfg).stem}.log 2>&1 '
            f'&& echo "✓ {cfg} done" >> /workspace/training.log '
            f'|| echo "✗ {cfg} FAILED" >> /workspace/training.log'
            for cfg in ALL_CONFIGS
        ]
        train_cmds = "\n".join(per_cfg_cmds) + '\necho "All experiments complete" | tee -a /workspace/training.log'
    else:
        cfg_stem = Path(config_path).stem
        train_cmds = (
            f'python scripts/04_experiment.py --config {config_path} '
            f'>> /workspace/training_{cfg_stem}.log 2>&1\n'
            f'echo "Training complete — exit $?" >> /workspace/training.log'
        )

    return f"""#!/bin/bash
set -e
echo "=== OpenScriptura Phase 2 Setup ===" | tee /workspace/setup.log

# System deps
apt-get update -qq && apt-get install -y -qq git wget curl 2>> /workspace/setup.log

# Clone repo
cd /workspace
git clone {GITHUB_REPO} openscriptura 2>> /workspace/setup.log
cd openscriptura

# Install Python deps
pip install -q -r requirements.txt --break-system-packages 2>> /workspace/setup.log
echo "✓ Dependencies installed" | tee -a /workspace/setup.log

# Dry-run validation
python scripts/04_experiment.py --config {config_path} --dry-run \
  2>&1 | tee -a /workspace/setup.log

echo "=== Starting training ===" | tee -a /workspace/setup.log

# Training (logs to separate file)
{train_cmds}
"""


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch(offer_id: int, config_path: str, all_configs: bool) -> int:
    """Launch an instance and return the instance ID."""
    onstart = _build_onstart(config_path, all_configs)

    cfg_label = "all-configs" if all_configs else Path(config_path).stem
    print(f"\nLaunching instance on offer {offer_id}...")
    print(f"  Image  : {DOCKER_IMAGE}")
    print(f"  Disk   : {DISK_GB}GB")
    print(f"  Config : {cfg_label}")
    print(f"  Repo   : {GITHUB_REPO}")

    rc, out = _vastai(
        "create", "instance", str(offer_id),
        "--image",       DOCKER_IMAGE,
        "--disk",        str(DISK_GB),
        "--ssh",
        "--direct",
        "--onstart-cmd", onstart,
        "--label",       f"openscriptura-{cfg_label}",
    )

    if rc != 0:
        print(f"ERROR launching instance:\n{out}")
        sys.exit(1)

    try:
        result = json.loads(out)
        instance_id = result["new_contract"]
    except (json.JSONDecodeError, KeyError):
        # Try to parse from text output
        import re
        m = re.search(r"new_contract.*?(\d+)", out)
        if m:
            instance_id = int(m.group(1))
        else:
            print(f"Launched but could not parse instance ID from:\n{out}")
            sys.exit(1)

    print(f"\n✓ Instance created: {instance_id}")
    return instance_id


# ---------------------------------------------------------------------------
# Status + SSH
# ---------------------------------------------------------------------------

def show_status(instance_id: int) -> None:
    rc, out = _vastai("show", "instance", str(instance_id), "--raw")
    if rc != 0:
        print(f"ERROR: {out}")
        return
    try:
        data = json.loads(out)
        if isinstance(data, list):
            data = data[0]
        status   = data.get("actual_status", "?")
        gpu      = data.get("gpu_name", "?")
        cost_hr  = data.get("dph_total", 0)
        cost_tot = data.get("total_flops", 0)
        print(f"\nInstance {instance_id}:")
        print(f"  Status : {status}")
        print(f"  GPU    : {gpu}")
        print(f"  Cost   : ${cost_hr:.3f}/hr")
    except Exception as e:
        print(f"Raw output:\n{out[:500]}")


def show_ssh(instance_id: int) -> None:
    rc, out = _vastai("ssh-url", str(instance_id))
    if rc != 0:
        print(f"ERROR getting SSH URL: {out}")
        return
    print(f"\nSSH command:")
    print(f"  {out.strip()}")
    print(f"\nMonitor training logs via SSH:")
    print(f"  tail -f /workspace/openscriptura/training_*.log")
    print(f"\nCheck setup log:")
    print(f"  cat /workspace/setup.log")


# ---------------------------------------------------------------------------
# Destroy
# ---------------------------------------------------------------------------

def destroy(instance_id: int) -> None:
    rc, out = _vastai("destroy", "instance", str(instance_id))
    if rc == 0:
        print(f"✓ Instance {instance_id} destroyed.")
    else:
        print(f"ERROR: {out}")


# ---------------------------------------------------------------------------
# Wait for running
# ---------------------------------------------------------------------------

def wait_for_running(instance_id: int, timeout_sec: int = 300) -> bool:
    """Poll until instance status == 'running' or timeout."""
    print(f"Waiting for instance {instance_id} to start", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout_sec:
        rc, out = _vastai("show", "instance", str(instance_id), "--raw")
        if rc == 0:
            try:
                data = json.loads(out)
                if isinstance(data, list):
                    data = data[0]
                status = data.get("actual_status", "")
                if status == "running":
                    print(" ✓ running")
                    return True
                elif status in ("exited", "deleted", "error"):
                    print(f"\nERROR: instance entered status '{status}'")
                    return False
            except Exception:
                pass
        print(".", end="", flush=True)
        time.sleep(15)
    print("\nTimeout waiting for instance to start.")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automate Phase 2 experiments on vast.ai RTX 4090."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--search",     action="store_true",
                       help="List top 10 cheapest RTX 4090 offers")
    group.add_argument("--config",     type=str,
                       help="Launch instance running this config (e.g. configs/exp_d.yaml)")
    group.add_argument("--status",     type=int, metavar="INSTANCE_ID",
                       help="Show status of a running instance")
    group.add_argument("--destroy",    type=int, metavar="INSTANCE_ID",
                       help="Destroy an instance")

    parser.add_argument("--offer-id",    type=int,
                        help="Use specific offer ID instead of cheapest available")
    parser.add_argument("--all-configs", action="store_true",
                        help="Run all 4 configs (A–D) sequentially on one instance")
    parser.add_argument("--wait",        action="store_true",
                        help="Wait for instance to reach 'running' before printing SSH")
    parser.add_argument("--top",         type=int, default=10,
                        help="Number of offers to show in --search (default: 10)")

    args = parser.parse_args()

    _check_vastai()

    if args.search:
        print("Searching for RTX 4090 offers...")
        offers = search_offers(top_n=args.top)
        if not offers:
            print("No offers found matching criteria.")
        else:
            print_offers(offers)
            print(f"To launch on the cheapest: --config configs/exp_d.yaml --offer-id {offers[0]['id']}")

    elif args.config:
        # Find offer
        if args.offer_id:
            offer_id = args.offer_id
            print(f"Using specified offer: {offer_id}")
        else:
            print("Searching for cheapest RTX 4090...")
            offers = search_offers(top_n=5)
            if not offers:
                print("No RTX 4090 offers found. Try --search to see what's available.")
                sys.exit(1)
            print_offers(offers)
            offer_id = offers[0]["id"]
            print(f"Selected cheapest offer: {offer_id} (${offers[0].get('dph_total',0):.3f}/hr)")

        instance_id = launch(offer_id, args.config, args.all_configs)

        if args.wait:
            running = wait_for_running(instance_id)
            if running:
                show_ssh(instance_id)
        else:
            print(f"\nInstance is starting. Check status in ~60 seconds:")
            print(f"  python scripts/vastai_run_experiments.py --status {instance_id}")
            print(f"\nOnce running, get SSH access:")
            print(f"  vastai ssh-url {instance_id}")
            print(f"\nTo destroy when done:")
            print(f"  python scripts/vastai_run_experiments.py --destroy {instance_id}")

    elif args.status:
        show_status(args.status)
        show_ssh(args.status)

    elif args.destroy:
        confirm = input(f"Destroy instance {args.destroy}? This cannot be undone. [y/N]: ")
        if confirm.lower() == "y":
            destroy(args.destroy)
        else:
            print("Cancelled.")


if __name__ == "__main__":
    main()
