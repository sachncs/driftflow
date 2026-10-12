"""Run a deterministic smoke test of the typed driftflow pipeline."""

import argparse
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driftflow import (
    CommonConfig,
    DriftModel,
    DriftState,
    DvsAdaptiveController,
    DVSSampler,
    GDSSApproximation,
    GraphState,
    GruMApproximation,
    LinearSchedule,
    decode_adjacency,
    get_dataset_config,
)


class SyntheticDampingDrift(DriftModel):
    """Simple autonomous drift that moves both modalities toward zero."""

    def evaluate(self, state: GraphState, time: float) -> DriftState:
        """Return ``-0.5 * state`` for both graph modalities."""
        del time
        return DriftState(
            tuple(tuple(-0.5 * value for value in row) for row in state.features),
            tuple(tuple(-0.5 * value for value in row) for row in state.adjacency),
        )


def make_random_graph(
    num_nodes: int, feature_dim: int, random_generator: random.Random
) -> GraphState:
    """Generate a reproducible random graph state."""
    features = [[random_generator.random() for _ in range(feature_dim)] for _ in range(num_nodes)]
    adjacency = [[random_generator.random() for _ in range(num_nodes)] for _ in range(num_nodes)]
    return GraphState.from_lists(features, adjacency)


def run_demo(
    model: str = "GruM",
    dataset: str = "QM9",
    solver: str = "Euler-Maruyama",
    seed: int = 42,
    use_approximation: bool = False,
) -> None:
    """Run a sample and print typed diagnostics."""
    common = CommonConfig()
    dataset_config = get_dataset_config(model, dataset)
    gamma = dataset_config.gamma_euler if solver == "Euler-Maruyama" else dataset_config.gamma_heun
    if gamma is None:
        raise ValueError(f"{solver} is not configured for {model}/{dataset}")

    if use_approximation:
        approximation_class = GruMApproximation if model == "GruM" else GDSSApproximation
        drift_model = approximation_class(num_nodes=5, feature_dim=3, seed=seed)
        print(f"Using {model} simplified approximation")
    else:
        drift_model = SyntheticDampingDrift()
        print("Using synthetic damping drift")

    sampler = DVSSampler(
        drift_model=drift_model,
        noise_schedule=LinearSchedule(sigma_min=0.01, sigma_max=0.5),
        controller=DvsAdaptiveController(common, dataset_config, gamma),
        solver_name=solver,
    )
    initial_state = make_random_graph(5, 3, random.Random(seed))
    result = sampler.sample(initial_state, terminal_time=1.0, seed=seed)
    steps = result.diagnostics.steps
    timesteps = [step.timestep for step in steps]
    decoded_adjacency = decode_adjacency([list(row) for row in result.state.adjacency])
    edge_count = sum(sum(row) for row in decoded_adjacency)

    print(f"Demo: model={model} dataset={dataset} solver={solver}")
    print(f"Sampling complete in {result.diagnostics.total_steps} steps")
    print(f"Final time: {result.diagnostics.final_time:.6f}")
    print(f"Mean dt: {statistics.mean(timesteps):.6f}")
    print(f"Min dt: {min(timesteps):.6f}  Max dt: {max(timesteps):.6f}")
    print(f"Decoded adjacency edges: {int(edge_count)}")


def main() -> None:
    """Parse CLI arguments and run the demo."""
    parser = argparse.ArgumentParser(description="Typed driftflow sampler demo")
    parser.add_argument("--model", default="GruM", choices=["GruM", "GDSS"])
    parser.add_argument("--dataset", default="QM9")
    parser.add_argument("--solver", default="Euler-Maruyama", choices=["Euler-Maruyama", "Heun"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-approximation", action="store_true")
    args = parser.parse_args()
    run_demo(args.model, args.dataset, args.solver, args.seed, args.use_approximation)


if __name__ == "__main__":
    main()
