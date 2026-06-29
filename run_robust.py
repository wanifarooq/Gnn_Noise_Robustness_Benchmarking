#!/usr/bin/env python3
"""Robust sweep runner — fault-isolated, never-dies dispatcher.

Runs each config of a sweep in its OWN child process (via `main.py --only-index N`)
under a wall-clock timeout, so that a single bad method (GPU OOM, a CPU hang on a
dense N x N op, or an OS OOM-kill of the child) can NEVER stall or kill the whole
sweep. Each outcome is logged and the runner moves on:

  [OK]          config produced experiment.json
  [OOM/FAIL]    child exited but wrote no result (OOM / crash / OS-kill of child)
  [TIMEOUT]     child exceeded --timeout and was killed (a hang)
  [SKIP-DONE]   experiment.json already present (resume)
  [SKIP-FAILED] a prior ROBUST_FAILED.json marker exists (use --retry-failed to retry)

The child is started in its own process group; on timeout the whole group is
killed (SIGTERM -> SIGKILL) so dataloader workers etc. die with it. The parent
only ever supervises children, so it is never at risk itself.

Usage:
  python run_robust.py -c configs/roman-empire.yaml --timeout 7200
  python run_robust.py -c configs/roman-empire_gcn_modified.yaml --timeout 14400
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import yaml

from sweep_utils import expand_yaml_sweeps, get_result_filename

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "main.py")
MARKER_NAME = "ROBUST_FAILED.json"


def _kill_group(proc):
    """Kill the child's whole process group: SIGTERM, then SIGKILL if it lingers."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def _write_marker(exp_dir, payload):
    try:
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, MARKER_NAME), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description="Robust, fault-isolated sweep runner")
    ap.add_argument("--config", "-c", required=True, help="Path to the sweep YAML")
    ap.add_argument("--results", "-r", default="results", help="Results base folder")
    ap.add_argument("--timeout", type=float, default=7200.0,
                    help="Per-config wall-clock timeout in seconds (default 7200=2h). "
                         "Must exceed the slowest LEGITIMATE config; a per-config "
                         "'timeout:' key in the YAML overrides this.")
    ap.add_argument("--num-runs", type=int, default=None, help="Override num_runs per config")
    ap.add_argument("--force", action="store_true", help="Re-run everything, ignore completed/failed")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Retry configs that have a ROBUST_FAILED.json marker")
    ap.add_argument("--cpu-fallback", action="store_true",
                    help="Re-enable main.py's CPU fallback on GPU OOM. Default OFF: a config "
                         "that OOMs is skipped immediately (no hours-long CPU crawl).")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    configs = expand_yaml_sweeps(config)
    total = len(configs)
    print(f"[robust] {total} configs from {args.config} | per-config timeout {args.timeout:.0f}s",
          flush=True)

    summary = {"ok": 0, "skip_done": 0, "skip_failed": 0, "oom_fail": 0, "timeout": 0}

    for i, cfg in enumerate(configs, 1):
        name = get_result_filename(cfg)
        exp_dir = os.path.join(args.results, name)
        result_json = os.path.join(exp_dir, "experiment.json")
        marker = os.path.join(exp_dir, MARKER_NAME)
        method = cfg.get("training", {}).get("method", "?")
        tag = f"[{i}/{total}] {name}"

        if os.path.exists(result_json) and not args.force:
            print(f"[SKIP-DONE]   {tag}", flush=True)
            summary["skip_done"] += 1
            continue
        if os.path.exists(marker) and not args.force and not args.retry_failed:
            print(f"[SKIP-FAILED] {tag} (prior failure; --retry-failed to retry)", flush=True)
            summary["skip_failed"] += 1
            continue
        # A fresh attempt: clear any stale failure marker.
        if os.path.exists(marker):
            try:
                os.remove(marker)
            except OSError:
                pass

        timeout_s = float(cfg.get("timeout", args.timeout))
        cmd = [sys.executable, MAIN, "--config", args.config,
               "--results", args.results, "--only-index", str(i)]
        if args.num_runs is not None:
            cmd += ["--num-runs", str(args.num_runs)]
        if args.force:
            cmd += ["--force"]
        if not args.cpu_fallback:
            cmd += ["--no-cpu-fallback"]

        print(f"[RUN]         {tag} | method={method} | timeout={timeout_s:.0f}s", flush=True)
        t0 = time.time()
        # start_new_session=True -> child is its own process-group leader, so we can
        # reliably kill it AND any workers it spawned without touching this parent.
        proc = subprocess.Popen(cmd, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            rc, timed_out = None, True
        except KeyboardInterrupt:
            _kill_group(proc)
            print("\n[robust] interrupted — killed current child, exiting.", flush=True)
            print(f"[robust] PARTIAL {summary}", flush=True)
            sys.exit(130)
        dur = time.time() - t0

        if os.path.exists(result_json):
            # Success is defined by a real result file, not the exit code — main.py's
            # own fail-isolation can exit 0 without writing a result.
            print(f"[OK]          {tag} | {dur:.0f}s", flush=True)
            summary["ok"] += 1
        elif timed_out:
            print(f"[TIMEOUT]     {tag} | killed after {dur:.0f}s (hang) — skipping", flush=True)
            _write_marker(exp_dir, {"reason": "timeout", "method": method,
                                    "duration_s": round(dur, 1), "timeout_s": timeout_s})
            summary["timeout"] += 1
        else:
            print(f"[OOM/FAIL]    {tag} | exit={rc} after {dur:.0f}s, no result — skipping",
                  flush=True)
            _write_marker(exp_dir, {"reason": "oom_or_crash", "method": method,
                                    "duration_s": round(dur, 1), "exit_code": rc})
            summary["oom_fail"] += 1

    print(f"[robust] DONE  ok={summary['ok']} timeout={summary['timeout']} "
          f"oom/fail={summary['oom_fail']} skip-done={summary['skip_done']} "
          f"skip-failed={summary['skip_failed']}  (of {total})", flush=True)


if __name__ == "__main__":
    main()
