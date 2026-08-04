import numpy as np
import pytest

from mdp import TabularMDP
from irl.maxent_linear import backward_pass
from sim.gridworld import build_gridworld_mdp, coord_to_state, state_to_coord
from data.simulate import generate_demonstrations


@pytest.fixture
def small_grid() -> tuple[TabularMDP, tuple[int, int], list[tuple[int, int]]]:
    width, height = 4, 4
    goal_pos = (3, 3)
    obstacles = [(1, 1)]
    mdp = build_gridworld_mdp(width, height, obstacles, goal_pos)
    return mdp, goal_pos, obstacles


def test_sampling_matches_backward_pass_distribution(small_grid):
    mdp, goal_pos, obstacles = small_grid
    theta = np.array([-1.0, 0.0])  # pousse vers le but (dist_goal, dist_obstacle)
    horizon = 6
    start_state = coord_to_state(0, 0, width=4)
    start_distribution = np.zeros(mdp.n_states)
    start_distribution[start_state] = 1.0

    action_probs, _V = backward_pass(mdp, theta, horizon)

    rng = np.random.default_rng(0)
    demos = generate_demonstrations(
        mdp, theta, start_distribution, horizon, n_demonstrations=5000, rng=rng
    )

    # fréquence empirique de la 1ère action réellement prise, déduite de s_0 -> s_1
    empirical_next_state = np.zeros(mdp.n_states)
    for demo in demos:
        empirical_next_state[demo[1]] += 1
    empirical_next_state /= len(demos)

    # distribution théorique du prochain état, dérivée de action_probs[0, start_state]
    theoretical_next_state = action_probs[0, start_state] @ mdp.transition[start_state]

    assert np.allclose(empirical_next_state, theoretical_next_state, atol=0.03)


def test_generated_trajectories_are_structurally_valid(small_grid):
    mdp, goal_pos, obstacles = small_grid
    theta = np.array([-1.0, 0.0])
    horizon = 6
    start_state = coord_to_state(0, 0, width=4)
    start_distribution = np.zeros(mdp.n_states)
    start_distribution[start_state] = 1.0

    rng = np.random.default_rng(1)
    demos = generate_demonstrations(
        mdp, theta, start_distribution, horizon, n_demonstrations=200, rng=rng
    )

    for demo in demos:
        assert len(demo) == horizon + 1
        assert np.all((demo >= 0) & (demo < mdp.n_states))
        for s_t, s_next in zip(demo[:-1], demo[1:]):
            assert np.any(mdp.transition[s_t, :, s_next] > 0), \
                f"transition {s_t}->{s_next} impossible selon mdp.transition"
        for s in demo:
            assert state_to_coord(s, width=4) not in obstacles