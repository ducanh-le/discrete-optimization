#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Pretty-print the content of a RCALBPLSolution.

The CP-SAT / DIDP logs only contain objective values and bounds: the actual
decisions (task -> workstation, start times per period, resource dispatch,
cycle time per period) live in the RCALBPLSolution object. This script dumps
them in a readable form, either from a pickle produced by run_pareto.py or by
re-solving an instance.

Usage:
    python print_solution.py <instance.json>
    python print_solution.py <instance.json> --pickle sol_xxx.pkl
"""
import argparse
import pickle
from pathlib import Path

from discrete_optimization.alb.rcalbp_l.parser import parse_rcalbpl_json
from discrete_optimization.alb.rcalbp_l.problem import RCALBPLProblem, RCALBPLSolution


def describe_solution(problem: RCALBPLProblem, sol: RCALBPLSolution) -> None:
    print("=" * 70)
    print(
        f"nb_tasks={problem.nb_tasks} nb_stations={problem.nb_stations} "
        f"nb_periods={problem.nb_periods} c_target={problem.c_target} "
        f"c_max={problem.c_max}"
    )

    # --- 1. Allocation: which task goes on which workstation (same for all periods)
    print("\n[Allocation task -> workstation]")
    for w in problem.stations:
        tasks_w = sorted(t for t in problem.tasks if sol.wks[t] == w)
        print(f"  WS {w}: tasks {tasks_w}")

    # --- 2. Resource dispatch: how much of each resource is put on each workstation
    print("\n[Resource dispatch (r, w) -> quantity]")
    for r in problem.resources:
        per_w = {w: sol.raw.get((r, w), 0) for w in problem.stations}
        print(
            f"  resource {r} (global capa {problem.capa_resources[r]}): "
            + ", ".join(f"WS{w}={q}" for w, q in per_w.items())
            + f"  | dispatched={sum(per_w.values())}"
        )

    # --- 3. Schedule per period
    real_cyc = problem.compute_actual_cycle_time_per_period(sol)
    print("\n[Schedule per period]")
    for p in problem.periods:
        kind = "unstable/fill-up" if p < problem.nb_stations else "stable"
        cost = (
            sol.cyc[p]
            if (p < problem.nb_stations or sol.cyc[p] > problem.c_target)
            else 0
        )
        print(
            f"\n  period {p} ({kind}): cyc_chosen={sol.cyc[p]} "
            f"cyc_used={real_cyc[p]} ramp_up_cost={cost}"
        )
        rows = []
        for t in problem.tasks:
            w = sol.wks[t]
            dur = problem.get_duration(t, p, w)
            st = sol.start.get((t, p), 0)
            rows.append((w, st, t, dur))
        for w, st, t, dur in sorted(rows):
            if dur == 0:
                print(f"    WS{w}  task {t}: not started yet (learning index < 0)")
            else:
                print(f"    WS{w}  task {t}: [{st}, {st + dur})  dur={dur}")

    # --- 4. Objectives / feasibility
    print("\n[Evaluation]")
    kpis = problem.evaluate(sol)
    for k, v in kpis.items():
        print(f"  {k}: {v}")
    print(f"  satisfy: {problem.satisfy(sol)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=str, help="path to the instance json")
    parser.add_argument(
        "--pickle",
        type=str,
        default=None,
        help="pickle dumped by run_pareto.py (dict with a 'sol' key)",
    )
    args = parser.parse_args()

    problem = parse_rcalbpl_json(Path(args.instance))

    if args.pickle is not None:
        with open(args.pickle, "rb") as f:
            sol = pickle.load(f)["sol"]
        sol.change_problem(problem)
    else:
        from discrete_optimization.alb.rcalbp_l.solvers import CpSatRCALBPLSolver
        from discrete_optimization.generic_tools.cp_tools import ParametersCp

        solver = CpSatRCALBPLSolver(problem=problem)
        solver.init_model(add_heuristic_constraint=False)
        res = solver.solve(
            parameters_cp=ParametersCp.default_cpsat(), time_limit=30
        )
        sol = res[-1][0]

    describe_solution(problem, sol)


if __name__ == "__main__":
    main()
