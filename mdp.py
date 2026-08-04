'''The datatype of an MDP as used in MaxEnt Linear'''
import numpy as np
from dataclasses import dataclass

@dataclass
class TabularMDP:
    n_states: int
    n_actions: int
    transition: np.ndarray   # shape (n_states, n_actions, n_states) — P(s'|s,a)
    features: np.ndarray     # shape (n_states, n_features) — f_s




def coord_to_state(row: int, col: int, width: int) -> int:
    return row * width + col

def state_to_coord(s: int, width: int) -> tuple[int, int]:
    return (s // width, s % width)