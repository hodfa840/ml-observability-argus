"""Single-command launcher for the Self-Healing ML system.

Starts API + Dashboard, waits for API to be ready, then runs drift simulation.

Usage:
    python run.py                        # gradual drift, 500 steps
    python run.py --drift sudden         # sudden drift
    python run.py --no-sim               # API + dashboard only (no simulation)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Always use the Python from the venv next to this file, regardless of how
# the script was invoked (so `python run.py` works without activating venv).
_venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
VENV_PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

API_URL = "http://localhost:8000"
DASH_URL = "http://localhost:8501"


def wait_for_api(timeout: int = 30) -> bool:
    import urllib.request
    print("  Waiting for API to start", end="", flush=True)
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"{API_URL}/health", timeout=2)
            print(" ready!")
            return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print(" TIMEOUT")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift", default="gradual",
                        choices=["gradual", "sudden", "seasonal", "mixed"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--no-sim", action="store_true",
                        help="Start API + dashboard only, skip simulation")
    args = parser.parse_args()

    procs = []

    print("\n========================================")
    print("  Self-Healing ML — Starting services")
    print("========================================\n")

    # 1. Start API
    print("[1/3] Starting FastAPI server...")
    api_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    procs.append(api_proc)

    if not wait_for_api():
        print("ERROR: API failed to start. Check logs.")
        for p in procs:
            p.terminate()
        sys.exit(1)

    # 2. Start dashboard
    print("[2/3] Starting Streamlit dashboard...")
    dash_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", "8501", "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    procs.append(dash_proc)
    time.sleep(3)  # give streamlit a moment

    print(f"\n  API docs  : {API_URL}/docs")
    print(f"  Dashboard : {DASH_URL}")
    print()

    # Open browser
    try:
        webbrowser.open(DASH_URL)
    except Exception:
        pass

    if args.no_sim:
        print("Services running. Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        # 3. Run simulation
        print(f"[3/3] Running drift simulation (type={args.drift}, steps={args.steps})...")
        print("      Dashboard will update every 10s — enable Auto-refresh in sidebar.\n")
        run = 1
        try:
            while True:
                drift_types = ["gradual", "sudden", "gradual", "seasonal"]
                drift = drift_types[(run - 1) % len(drift_types)]
                print(f"  Simulation run #{run} (drift={drift})...")
                subprocess.run(
                    [VENV_PYTHON, "scripts/simulate_drift.py",
                     "--drift-type", drift,
                     "--steps", str(args.steps),
                     "--delay", "0.05",
                     "--feedback-lag", "5"],
                    cwd=ROOT,
                    check=True,
                )
                run += 1
        except KeyboardInterrupt:
            pass
        except subprocess.CalledProcessError as e:
            print(f"Simulation error: {e}")

    print("\nShutting down...")
    for p in procs:
        p.terminate()


if __name__ == "__main__":
    main()
