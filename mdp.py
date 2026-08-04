'''The datatype of an MDP as used in MaxEnt Linear'''
import numpy as np
from dataclasses import dataclass

@dataclass
class TabularMDP:
    n_states: int
    n_actions: int
    transition: np.ndarray   # shape (n_states, n_actions, n_states) — P(s'|s,a)
    features: np.ndarray     # shape (n_states, n_features) — f_s