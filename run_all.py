#!/usr/bin/env python3
"""Единая точка входа в репозиторий.

Шаги:
  pipeline     финальный пайплайн (7 окон → out/clusters_all.csv, out/jaccard.csv, out/compare_final.csv)
  metrics      ICVI-метрики → out/metrics.csv
  validate     валидация: baselines, methods, official (3 шага)
  experiments  все эксперименты из отчёта, по порядку зависимостей
  viz          лендинг: index_basket.html → visualization/site/
  smoke        смок-тест на 25 МО (быстрая проверка окружения)
  all          pipeline → metrics → validate → experiments → viz

Управление прогоном:
  python run_all.py all                  # полный прогон
  python run_all.py all --resume         # пропустить шаги, уже ok в этом прогоне
  python run_all.py experiments --resume # продолжить после падения
  python run_all.py --list               # шаги + их статус из последнего прогона
  python run_all.py all --dry-run        # показать план (что запустится, что пропустится)
  python run_all.py all --quiet          # не стримить вывод дочерних процессов (только логи)
  python run_all.py --forget             # очистить состояние (все шаги станут «не запущены»)
  python run_all.py --forget pipeline    # сбросить один шаг

Каждый шаг: вывод стримится в консоль И пишется в out/logs/<шаг>.log;
состояние — out/.run_state.json (out/ в git не идёт). При ошибке шаг
помечается failed, дальше выполнение не идёт — почините и --resume.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
LOGS = OUT / "logs"
STATE = OUT / ".run_state.json"

PY = sys.executable

EXPERIMENTS = [
    "features_exp", "features_exp2", "features_exp3", "features_exp4",  # G3: types_profile_g3
    "basket_index",  # → out/basket_index_mo.csv
    "paper_exp1_gravity", "paper_exp2_pp", "paper_exp3_regionalization",
    "paper_exp4_cvi", "paper_exp5_migration", "paper_exp6_kmeans", "paper_exp7_gravity",
    "price_strat", "price_ineq", "qol_stability", "review2", "design",
    "tune",
    "basket_cluster",  # читает out/basket_index_mo.csv
    "basket_final",  # читает out/types_profile_g3.csv → пишет out/basket_economy_clusters.csv
    "type_aggregates",  # читает out/basket_economy_clusters.csv
]


def step_cmd(name: str) -> list[str]:
    if name == "pipeline":
        return [PY, "-m", "scripts.clustering.run"]
    if name == "metrics":
        return [PY, "-m", "scripts.clustering.metrics"]
    if name == "smoke":
        return [PY, "-m", "scripts.clustering.smoke"]
    if name == "viz":
        return [PY, "visualization/build_landing.py"]
    if name == "tune":
        return [PY, "-m", "scripts.clustering.tune"]
    if name.startswith(("baselines", "methods", "official")):
        return [PY, "-m", f"scripts.clustering.validate.{name}"]
    if name in EXPERIMENTS:
        return [PY, "-m", f"scripts.clustering.experiments.{name}"]
    raise ValueError(f"неизвестный шаг: {name}")


def expand(step: str) -> list[str]:
    if step == "validate":
        return ["baselines", "methods", "official"]
    if step == "experiments":
        return EXPERIMENTS
    if step == "all":
        return ["pipeline", "metrics", "baselines", "methods", "official",
                *EXPERIMENTS, "viz"]
    return [step]


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"run": None, "steps": {}}


def save_state(state: dict) -> None:
    OUT.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def run_step(name: str, quiet: bool, state: dict) -> bool:
    cmd = step_cmd(name)
    LOGS.mkdir(exist_ok=True)
    log = open(LOGS / f"{name}.log", "w", encoding="utf-8")
    log.write(f"$ {' '.join(cmd)}\n{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n")
    log.flush()
    t0 = time.time()
    print(f"  $ {' '.join(cmd)}")
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, cwd=ROOT, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            if not quiet:
                sys.stdout.write(line)
                sys.stdout.flush()
            log.write(line)
        rc = p.wait()
    except (OSError, KeyboardInterrupt) as e:
        rc, tail = 1, str(e)
    else:
        tail = None
    dt = time.time() - t0
    ok = rc == 0
    state["steps"][name] = {
        "status": "ok" if ok else "failed",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed": round(dt, 1),
    }
    save_state(state)
    mark = "ok" if ok else "FAIL"
    print(f"  {name}: {mark} ({dt:.0f} c)  [лог: out/logs/{name}.log]")
    if not ok:
        logf = LOGS / f"{name}.log"
        if tail is not None:
            print(f"  ошибка запуска: {tail}")
        lines = logf.read_text(encoding="utf-8").splitlines()
        print(f"  --- последние строки out/logs/{name}.log ---")
        for line in lines[-25:]:
            print(f"  {line}")
        print(f"  исправьте проблему и продолжите: python run_all.py ... --resume")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", default=["all"],
                    help="шаги (по умолчанию all); возможны: " +
                    "pipeline metrics validate experiments viz smoke " +
                    "baselines methods official + отдельные эксперименты")
    ap.add_argument("--resume", action="store_true",
                    help="пропустить шаги со статусом ok в out/.run_state.json")
    ap.add_argument("--dry-run", action="store_true", help="показать план и выйти")
    ap.add_argument("--list", action="store_true", help="таблица шагов + статусы и выйти")
    ap.add_argument("--forget", nargs="*", metavar="STEP",
                    help="сбросить состояние (все шаги или перечисленные) и выйти")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="не стримить вывод дочерних процессов (пишется в out/logs/)")
    args = ap.parse_args()

    state = load_state()
    if args.forget is not None:
        if args.forget:
            for s in args.forget:
                state["steps"].pop(s, None)
        else:
            state = {"run": None, "steps": {}}
        save_state(state)
        print(f"состояние сброшено: {', '.join(args.forget) if args.forget else 'все шаги'}")
        return 0

    if args.list:
        print(f"{'шаг':<22} {'статус':<8} {'время':>7}  когда")
        for name in expand("all"):
            rec = state["steps"].get(name, {})
            print(f"{name:<22} {rec.get('status', '-'):<8} "
                  f"{str(rec.get('elapsed', '-')):>7}  {rec.get('ts', '')}")
        return 0

    plan = []
    for step in args.steps:
        plan.extend(expand(step))
    plan = list(dict.fromkeys(plan))  # дедупликация с сохранением порядка

    if args.dry_run:
        for name in plan:
            rec = state["steps"].get(name)
            skip = args.resume and rec and rec["status"] == "ok"
            print(f"  {'skip ' if skip else 'run   '} {name}")
        return 0

    state["run"] = state["run"] or datetime.now(timezone.utc).isoformat(timespec="seconds")
    t_total = time.time()
    done = 0
    for name in plan:
        rec = state["steps"].get(name)
        if args.resume and rec and rec["status"] == "ok":
            print(f"[{done + 1}/{len(plan)}] {name}: skip (ok, {rec['elapsed']} c)")
            done += 1
            continue
        print(f"[{done + 1}/{len(plan)}] {name} ...", flush=True)
        if not run_step(name, args.quiet, state):
            return 1
        done += 1

    dt = time.time() - t_total
    print(f"\nготово: {done}/{len(plan)} шагов за {dt / 60:.1f} мин "
          f"(состояние: out/.run_state.json, логи: out/logs/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
