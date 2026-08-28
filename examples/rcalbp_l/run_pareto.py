import logging
import pickle
import time
from matplotlib import pyplot as plt

from pathlib import Path
import argparse
import sys
import ast
import faulthandler
faulthandler.enable()

from discrete_optimization.alb.rcalbp_l import get_data_available, parse_rcalbpl_json
from discrete_optimization.alb.rcalbp_l.solvers import (
    BackwardSequentialRCALBPLSolver,
    CpSatRCALBPLSolver,
)
from discrete_optimization.alb.rcalbp_l.solvers.pareto_postprocess import (
    DpRCALBPLPostProSolver,
    RampUpParetoSolverPostpro,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.pareto_tools import CpsatParetoSolver

logging.basicConfig(level=logging.INFO)


def run_pareto(instance_path, future_chunk_size=1, phase2_chunk_size=2, time_limit_phase1=300, time_limit_phase2=60, use_sgs_warm_start=True, time_limit=6300):
    instance = instance_path.name
    problem = parse_rcalbpl_json(instance_path)
    # problem.nb_periods = 5
    # problem.periods = range(problem.nb_periods)
    from discrete_optimization.generic_tools.sequential_metasolver import (
        SequentialMetasolver,
        SubBrick,
    )

    p = ParametersCp.default_cpsat()
    p.nb_process = 20
    brick1 = SubBrick(
        BackwardSequentialRCALBPLSolver,
        kwargs=dict(
            future_chunk_size=future_chunk_size,
            phase2_chunk_size=phase2_chunk_size,
            time_limit_phase1=time_limit_phase1,
            time_limit_phase2=time_limit_phase2,
            use_sgs_warm_start=use_sgs_warm_start,
            parameters_cp=p,
            ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        ),
    )
    brick2 = SubBrick(
        CpSatRCALBPLSolver,
        dict(
            add_heuristic_constraint=False,
            parameters_cp=p,
            ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
            time_limit=time_limit,
        ),
    )
    solver = SequentialMetasolver(list_subbricks=[brick1, brick2], problem=problem)
    res = solver.solve()
    sol = res[-1][0]
    res_dict = {"instance": instance, "sol": sol}
    pickle.dump(
        res_dict, open(f"sol_{instance}_dp_more_{time.process_time()}.pkl", "wb")
    )
    import didppy as dp

    postpro_solver = DpRCALBPLPostProSolver(problem=problem)
    front = postpro_solver.create_result_storage([])
    postpro_solver.init_model(from_solution=sol, max_nb_adjustments=1)
    for i in range(1, len(postpro_solver.decision_step) + 1):
        postpro_solver.init_model(from_solution=sol, max_nb_adjustments=i)
        res = postpro_solver.solve(solver=dp.CABS, time_limit=60, threads=20)
        front.extend(res[-1:])
        print(problem.evaluate(res[-1][0]))

    f1s, f2s = [], []
    for sol, fit in front:
        eval_ = problem.evaluate(sol)
        dur_rampup = eval_["ramp_up_duration"]
        nb_adjustments = eval_["nb_adjustments"]
        print(f"  Obj: {fit} | Sol: {sol}")
        if nb_adjustments >= 1:
            f1s.append(nb_adjustments)
            f2s.append(dur_rampup)
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(f1s, f2s, c="green", s=100, label="Pareto Front")
    # Known optima for Example 9 are (1, 2) and (3, 0)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Pareto Front (Epsilon Constraint via Add/Remove)")
    plt.grid(True)
    plt.legend()
    plt.savefig(Path(".") / "res" / "fig" /
                f"{instance[:-5]}_{future_chunk_size}_{phase2_chunk_size}_{time_limit_phase1}_{time_limit_phase2}_{use_sgs_warm_start}_{time_limit}.png")
    print(problem.evaluate(sol), problem.satisfy(sol))
    # plt.show()
    # fig, slider = plot_rcalbpl_dashboard(problem, sol)


def main_pareto(instance="187_2_26_2880.json"):
    problem = parse_rcalbpl_json(instance)
    # problem.nb_periods = 5
    # problem.periods = range(problem.nb_periods)
    from discrete_optimization.generic_tools.sequential_metasolver import (
        SequentialMetasolver,
        SubBrick,
    )

    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    brick1 = SubBrick(
        BackwardSequentialRCALBPLSolver,
        kwargs=dict(
            future_chunk_size=1,
            phase2_chunk_size=5,
            time_limit_phase1=200,
            time_limit_phase2=50,
            use_sgs_warm_start=True,
            parameters_cp=p,
            ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        ),
    )
    brick2 = SubBrick(
        CpSatRCALBPLSolver,
        dict(
            add_heuristic_constraint=False,
            parameters_cp=p,
            ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
            time_limit=500,
        ),
    )
    solver = SequentialMetasolver(list_subbricks=[brick1, brick2], problem=problem)
    res = solver.solve()
    sol = res[-1][0]
    res_dict = {"instance": instance, "sol": sol}
    pickle.dump(
        res_dict, open(f"sol_{instance}_cp_more_{time.process_time()}.pkl", "wb")
    )

    postpro_solver = RampUpParetoSolverPostpro(problem=problem)
    postpro_solver.init_model(from_solution=sol)
    pareto_solver = CpsatParetoSolver(
        problem=problem,
        solver=postpro_solver,
        objective_names=["change_cost", "ramp_up_cost"],
        dict_function={
            "change_cost": lambda sol: problem.evaluate(sol)["nb_adjustments"],
            "ramp_up_cost": lambda sol: problem.evaluate(sol)["ramp_up_duration"],
        },
        delta_ref_improvement=[0, 0],
        delta_abs_improvement=[1, 1],
    )
    front = pareto_solver.solve(
        obj_vars=[
            postpro_solver.variables["objectives"][c]
            for c in pareto_solver.objective_names
        ],
        time_limit=100,
        subsolver_kwargs={
            "time_limit": 4,
            "parameters_cp": ParametersCp.default_cpsat(),
        },
    )
    f1s, f2s = [], []
    for sol, fit in front:
        print(f"  Obj: {fit} | Sol: {sol}")
        if pareto_solver.dict_function["change_cost"](sol) >= 1:
            f1s.append(pareto_solver.dict_function["change_cost"](sol))
            f2s.append(pareto_solver.dict_function["ramp_up_cost"](sol))
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(f1s, f2s, c="green", s=100, label="Pareto Front")
    # Known optima for Example 9 are (1, 2) and (3, 0)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Pareto Front (Epsilon Constraint via Add/Remove)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"pareto_rcsalbp_{instance[:-5]}_more.png")
    print(problem.evaluate(sol), problem.satisfy(sol))
    # plt.show()
    # fig, slider = plot_rcalbpl_dashboard(problem, sol)


def main_pareto_dp(instance="187_2_26_2880.json"):
    problem = parse_rcalbpl_json(instance)
    # problem.nb_periods = 5
    # problem.periods = range(problem.nb_periods)
    from discrete_optimization.generic_tools.sequential_metasolver import (
        SequentialMetasolver,
        SubBrick,
    )

    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    brick1 = SubBrick(
        BackwardSequentialRCALBPLSolver,
        kwargs=dict(
            future_chunk_size=1,
            phase2_chunk_size=5,
            time_limit_phase1=200,
            time_limit_phase2=50,
            use_sgs_warm_start=True,
            parameters_cp=p,
            ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        ),
    )
    brick2 = SubBrick(
        CpSatRCALBPLSolver,
        dict(
            add_heuristic_constraint=False,
            parameters_cp=p,
            ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
            time_limit=500,
        ),
    )
    solver = SequentialMetasolver(list_subbricks=[brick1, brick2], problem=problem)
    res = solver.solve()
    sol = res[-1][0]
    res_dict = {"instance": instance, "sol": sol}
    pickle.dump(
        res_dict, open(f"sol_{instance}_dp_more_{time.process_time()}.pkl", "wb")
    )
    import didppy as dp

    postpro_solver = DpRCALBPLPostProSolver(problem=problem)
    front = postpro_solver.create_result_storage([])
    postpro_solver.init_model(from_solution=sol, max_nb_adjustments=1)
    for i in range(1, len(postpro_solver.decision_step) + 1):
        postpro_solver.init_model(from_solution=sol, max_nb_adjustments=i)
        res = postpro_solver.solve(solver=dp.CABS, time_limit=5, threads=10)
        front.extend(res[-1:])
        print(problem.evaluate(res[-1][0]))

    f1s, f2s = [], []
    for sol, fit in front:
        eval_ = problem.evaluate(sol)
        dur_rampup = eval_["ramp_up_duration"]
        nb_adjustments = eval_["nb_adjustments"]
        print(f"  Obj: {fit} | Sol: {sol}")
        if nb_adjustments >= 1:
            f1s.append(nb_adjustments)
            f2s.append(dur_rampup)
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(f1s, f2s, c="green", s=100, label="Pareto Front")
    # Known optima for Example 9 are (1, 2) and (3, 0)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Pareto Front (Epsilon Constraint via Add/Remove)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"pareto_rcsalbp_{instance[:-5]}_dp_more.png")
    print(problem.evaluate(sol), problem.satisfy(sol))
    # plt.show()
    # fig, slider = plot_rcalbpl_dashboard(problem, sol)


def main_script():
    # main_pareto_dp("187_2_26_2880.json")
    for instance in get_data_available():
        main_pareto_dp(instance)
        main_pareto(instance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inst_name', type=str, help='instance name without .json')
    parser.add_argument('inst_reduced', type=str, help='type of reduced instance (empty string / ws / t)')
    parser.add_argument('future_chunk_size', type=int, help='')
    parser.add_argument('phase2_chunk_size', type=int, help='')
    parser.add_argument('time_limit_phase1', type=int, help='')
    parser.add_argument('time_limit_phase2', type=int, help='')
    parser.add_argument('use_sgs_warm_start', type=int, help='')
    parser.add_argument('time_limit', type=int, help='')
    args = parser.parse_args()
    (inst_name, inst_reduced, future_chunk_size, phase2_chunk_size, time_limit_phase1, time_limit_phase2, use_sgs_warm_start, time_limit) = (
        args.inst_name, args.inst_reduced,
        args.future_chunk_size, args.phase2_chunk_size, args.time_limit_phase1, args.time_limit_phase2, bool(args.use_sgs_warm_start), args.time_limit
    )

    didactic = bool(ast.literal_eval(input('Do you want to run didactic instance ? (1 / 0) : ')))
    if didactic:
        inst_name = "6_2_6_12"
        inst_path = Path("..") / ".." / ".." / "these-ONERA" / "data" / "instances" / "didactic" / "6_2_6_12.json"
    else:
        if not inst_reduced:
            inst_type = 'airplane'
        else:
            inst_type = f'airplane-{inst_reduced}Reduced'
        inst_path = Path("..") / ".." / ".." / "these-ONERA" / "data" / "instances" / inst_type / f"{inst_name}.json"

    log_path = (Path(".") / "res" / "log" /
                f"{inst_name}_{future_chunk_size}_{phase2_chunk_size}_{time_limit_phase1}_{time_limit_phase2}_{use_sgs_warm_start}_{time_limit}.log")

    print(f"Running meta_solvers on {inst_path.name}")
    print("...")
    with open(log_path, 'w') as logfile:
        sys.stdout = logfile
        run_pareto(inst_path, future_chunk_size, phase2_chunk_size, time_limit_phase1, time_limit_phase2, use_sgs_warm_start, time_limit)
        sys.stdout = sys.__stdout__
    print(f"Finished meta_solvers on {inst_path.name}")
