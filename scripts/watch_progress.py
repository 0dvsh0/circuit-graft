#!/usr/bin/env python3
"""Watch running CircuitGraft stages and report progress + ETA.

Read-only: parses the stage logs under D:\\cg_tmp (stdout in *.log, tqdm
progress bars land in *.err) and the overall run plan in D:\\cg_tmp\\plan.json.
Never touches the GPU, so it is safe to run in a second terminal while a stage
runs.

Usage:
  python scripts/watch_progress.py                    # auto-detect newest log
  python scripts/watch_progress.py --log D:\\cg_tmp\\s2_drift.err
  python scripts/watch_progress.py --watch 30         # refresh every 30 s
  python scripts/watch_progress.py --plan             # overall budget view
  python scripts/watch_progress.py --tail 15          # show last 15 raw lines
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

LOG_DIR = Path(r"D:\cg_tmp")
PLAN_PATH = LOG_DIR / "plan.json"

# tqdm bar:  45%|xxxx | 4500/10000 [5:55:12<7:13:00, 4.74s/it]
TQDM_RE = re.compile(
    r"(?P<pct>\d+)%\|[^\n|]*\|\s*(?P<done>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[^\]<]+)<(?P<remaining>[^\],]+),\s*(?P<rate>[^\]]+)\]"
)
# transformers Trainer log line: {'loss': 2.9, ..., 'epoch': 0.01}
TRAIN_LOG_RE = re.compile(r"\{'loss':[^}]*'epoch':\s*([0-9.eE+-]+)")


def _last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    last = None
    for m in pattern.finditer(text):
        last = m
    return last


def parse_log(path: Path) -> dict[str, object] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    age_s = time.time() - path.stat().st_mtime
    tqdm = _last_match(TQDM_RE, text)
    if tqdm:
        return {
            "source": str(path),
            "age_s": round(age_s, 1),
            "kind": "tqdm",
            "done": int(tqdm.group("done")),
            "total": int(tqdm.group("total")),
            "pct": int(tqdm.group("pct")),
            "elapsed": tqdm.group("elapsed").strip(),
            "remaining": tqdm.group("remaining").strip(),
            "rate": tqdm.group("rate").strip(),
        }
    train = _last_match(TRAIN_LOG_RE, text)
    if train:
        epoch = float(train.group(1))
        # S2 drift: epoch counts optimizer epochs (0..2); 5000 rows/epoch.
        step = int(round(epoch * 5000))
        return {
            "source": str(path),
            "age_s": round(age_s, 1),
            "kind": "trainer-epoch",
            "done": step,
            "total": 10000,
            "pct": int(100 * step / 10000),
            "epoch": epoch,
        }
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return {
        "source": str(path),
        "age_s": round(age_s, 1),
        "kind": "tail",
        "last_line": lines[-1][:200] if lines else "(empty)",
    }


def newest_logs() -> list[Path]:
    if not LOG_DIR.exists():
        return []
    files = sorted(
        (p for p in LOG_DIR.iterdir() if p.suffix in (".log", ".err") and p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[:8]


def report(auto: bool, explicit: Path | None) -> None:
    candidates = [explicit] if explicit else newest_logs()
    # prefer .err files (tqdm) but consider everything
    for path in candidates[:4]:
        status = parse_log(path)
        if not status:
            continue
        if status["kind"] == "tail" and auto and len(candidates) > 1:
            continue
        print(format_status(status))


def format_status(status: dict[str, object]) -> str:
    kind = status["kind"]
    src = Path(str(status["source"])).name
    if kind in ("tqdm", "trainer-epoch"):
        return (
            f"[{src}] {status['done']}/{status['total']} steps "
            f"({status['pct']}%)"
            + (f" elapsed {status['elapsed']}" if status.get("elapsed") else "")
            + (f" remaining {status['remaining']}" if status.get("remaining") else "")
            + (f" rate {status['rate']}" if status.get("rate") else "")
            + f"  (log updated {status['age_s']}s ago)"
        )
    return f"[{src}] {status.get('last_line', '')}  (log updated {status['age_s']}s ago)"


def show_plan() -> None:
    if not PLAN_PATH.exists():
        print(f"no plan file at {PLAN_PATH}")
        return
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    budget = float(plan.get("budget_hours", 36))
    started = plan.get("started_at")
    stages = plan.get("stages", [])
    done_h = sum(float(s.get("spent_hours", 0)) for s in stages if s.get("status") == "done")
    remain_h = sum(float(s.get("eta_hours", 0)) for s in stages if s.get("status") != "done")
    elapsed_h = 0.0
    if started:
        elapsed_h = (time.time() - time.mktime(time.strptime(started, "%Y-%m-%d %H:%M"))) / 3600
    print(f"budget: {budget:.1f} h   elapsed since start: {elapsed_h:.1f} h")
    print(f"stages done so far: {done_h:.1f} h   projected remaining: {remain_h:.1f} h")
    projected = elapsed_h + remain_h
    flag = "OK" if projected <= budget else "OVER BUDGET"
    print(f"projected total: {projected:.1f} h vs {budget:.1f} h budget -> {flag}")
    print()
    for s in stages:
        mark = {"done": "x", "running": ">", "todo": " "}.get(str(s.get("status")), "?")
        eta = s.get("eta_hours", 0)
        spent = s.get("spent_hours", 0)
        print(
            f"[{mark}] {s.get('name', '?'):28s}"
            + (f" done in {float(spent):.1f} h" if s.get("status") == "done" else f" eta {float(eta):.1f} h")
            + (f"  - {s['note']}" if s.get("note") else "")
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=None, help="explicit log/err file")
    parser.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    parser.add_argument("--plan", action="store_true", help="show overall budget plan")
    parser.add_argument("--tail", type=int, default=0, help="show last N raw lines of the newest log")
    args = parser.parse_args()

    if args.plan:
        show_plan()
        return 0
    if args.tail:
        paths = [args.log] if args.log else newest_logs()[:1]
        for p in paths:
            if p and p.exists():
                print(f"--- {p} ---")
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                for line in lines[-args.tail:]:
                    print(line[:300])
        return 0

    if args.watch:
        try:
            while True:
                print(time.strftime("%H:%M:%S"), " ", end="")
                report(auto=True, explicit=args.log)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0
    report(auto=True, explicit=args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
