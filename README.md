# Swim-IRL

Inverse reinforcement learning (IRL) on real trajectories of **active**
microscopic swimmers, to infer the reward function under which their
behavior is *as if* optimal.

## Motivation

Active microscopic swimmers — bacteria, migrating cells, nematodes — clearly
behave in a directed, purposeful way in response to chemical gradients, but
we rarely have direct access to "the" objective driving that behavior. IRL
offers a principled way to ask: what is the simplest reward function under
which the observed behavior is (quasi-)optimal? Swim-IRL builds that
pipeline from scratch, starting with a fully simulated, controlled testbed
before ever touching real data.

## Objective

> Find the simplest reward function under which the observed behavior is
> quasi-optimal, then test whether that reward is (a) interpretable,
> (b) predictive out-of-sample, (c) consistent with the known
> biochemical/neural mechanism.

This is **not** "discovering the animal's true reward." It is an explicit
*as-if* model. IRL is ill-posed — several different reward functions can
explain the same trajectories — and this project is meant to handle that
ambiguity, not hide it.

## Target organism

***C. elegans*** — chemotaxis. The literature already decomposes navigation
into two parallel mechanisms:
- **pirouette** (klinokinesis): sharp reorientations, triggered when the
  perceived concentration *drops* over time;
- **weathervane** (klinotaxis): gradual curving of the heading toward the
  gradient.

This is a good test case for IRL: biology has already proposed a two-channel
behavioral structure — IRL should either recover it from behavior alone, or
give a clear reason why it doesn't.

## Approach

- **Linear MaxEnt IRL** (Ziebart, Maas, Bagnell & Dey, AAAI 2008) first —
  prioritizing interpretability, over reward features that are isolable and
  testable one at a time. Validated on a small tabular gridworld before
  anything else (Phase 0a).
- **Deep MaxEnt IRL** (Wulfmeier, Ondruska & Posner, 2015 — see References)
  extends linear MaxEnt to a nonlinear (neural) reward, but keeps the same
  exact tabular DP — which doesn't scale to a continuous environment like
  NanoGoal-RL. So in practice it's implemented here as **Guided Cost
  Learning** (Finn, Levine & Abbeel, ICML 2016 — see References): the
  sampling-based descendant of the same idea, usable on continuous state.
  A reward network is trained via importance-sampled maximum-entropy IRL.
  Policy optimization is handed off to an existing RL library
  (Stable-Baselines3 / PPO), so the effort goes into the reward-learning
  loop itself, not into reimplementing policy gradients.
- **AIRL** (Fu, Luo & Levine, 2018) via the `imitation` library
  (HumanCompatibleAI), for a more transferable, dynamics-independent
  reward over continuous state.
- Both continuous-state methods (Deep MaxEnt / AIRL) are validated on a
  second synthetic testbed before touching real data: the continuous,
  known-reward environment from
  [NanoGoal-RL v2](https://github.com/Josh012006/NanoGoal-RL/tree/v2),
  reused here as a controlled continuous-control sanity check, independent
  of the worm data (Phase 0b).
- Core difficulty: the worm's decisions are not observed directly, only its
  position/posture over time → heading, speed, and reorientation events
  (pirouette vs. weathervane) will need to be reconstructed from real
  trajectories before IRL can be run (Phase 2).

## References

Key papers this project builds on — read before implementing the
corresponding phase.

- [x] Ziebart, B. D., Maas, A., Bagnell, J. A., & Dey, A. K. (2008).
  *Maximum Entropy Inverse Reinforcement Learning*. AAAI.
  [PDF](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf)
  Core method for Phase 0a — resolves the ambiguity of choosing among reward
  functions that equally explain the observed behavior, via the principle
  of maximum entropy.
- [ ] Wulfmeier, M., Ondruska, P., & Posner, I. (2015). *Maximum Entropy
  Deep Inverse Reinforcement Learning*.
  [arXiv:1507.04888](https://arxiv.org/abs/1507.04888)
  Extends Ziebart 2008 to a nonlinear (neural) reward, still via exact
  tabular DP. Origin of the term "Deep MaxEnt IRL" used in this project —
  see the Finn et al. 2016 entry below for the continuous-state version
  actually implemented (Phase 0b, Phase 1).
- [ ] Finn, C., Levine, S., & Abbeel, P. (2016). *Guided Cost Learning: Deep
  Inverse Optimal Control via Policy Optimization*. ICML.
  [arXiv:1603.00448](https://arxiv.org/abs/1603.00448)
  Sample-based descendant of Wulfmeier 2015, usable on continuous
  state/action spaces — this is what "Deep MaxEnt IRL" concretely means in
  this project (Phase 0b, Phase 1).
- [ ] Fu, J., Luo, K., & Levine, S. (2018). *Learning Robust Rewards with
  Adversarial Inverse Reinforcement Learning*. ICLR.
  [arXiv:1710.11248](https://arxiv.org/abs/1710.11248)
  Planned extension (Phase 1+) — decouples the recovered reward from
  environment dynamics for better transfer across conditions, relevant to
  Phase 4.
- [ ] Vergassola, M., Villermaux, E., & Shraiman, B. I. (2007). *'Infotaxis'
  as a strategy for searching without gradients*. Nature, 445(7126), 406–409.
  [Nature](https://www.nature.com/articles/nature05464) ·
  [free PDF](https://faculty.washington.edu/minster/bio_inspired_robotics/research_articles/vergassola_vellermaux_shraiman_infotaxis_searching_without_gradients_nature2007.pdf)
  Basis for the candidate "information" reward feature (Phase 1).

## Roadmap (phases)

**Phase 0a — Tabular simulation validation (blocking).**
Build a small discretized/tabular simulated swimmer with a CHOSEN
ground-truth reward, verify that linear MaxEnt IRL recovers it, on ≥ 3
different test rewards.

**Phase 0b — Continuous simulation validation (blocking).**
Same validation, one level up in complexity: verify that Deep MaxEnt IRL
(Guided Cost Learning) and AIRL each recover a sensible reward on a
continuous-control environment with a known ground-truth reward, before
either method ever sees worm data. Testbed:
[NanoGoal-RL v2](https://github.com/Josh012006/NanoGoal-RL/tree/v2),
reused as an external, already-known-reward environment (not
reimplemented). Real data is off limits until **both** Phase 0a and Phase
0b are met.

**Phase 1 — State/action representation + features.**
Candidate state: local concentration, estimated local gradient, temporal
derivative of concentration, speed, heading. Candidate features:
concentration, velocity-gradient alignment, temporal derivative
(chemotaxis hypothesis), effort cost, information term (infotaxis,
Vergassola 2007 — see References).
**Open decision**: a unified action space (a single turning angle of
variable magnitude) vs. two explicit channels (continuous for weathervane +
discrete for pirouette, closer to the biology)? To be settled after reading
Ziebart 2008, before writing the simulator.

**Phase 2 — Real data.**
Load *C. elegans* trajectories in a gradient (NaCl or odorant). Cleaning,
interpolation, speed estimation, pirouette/weathervane segmentation.
Not allowed: passive (purely Brownian) particles → degenerate reward.

**Phase 3 — Comparison to the known mechanism.**
For the linear model: does the "temporal derivative" feature emerge as
dominant, consistent with what is known about the real sensory mechanism?
For Deep MaxEnt / AIRL, the reward is no longer a transparent linear sum,
so mechanism-consistency is tested via feature ablation / sensitivity
analysis instead of reading off a weight directly. This phase also
compares interpretability and mechanism-consistency across all three
recovered rewards (linear MaxEnt, Deep MaxEnt, AIRL).

**Phase 4 — Robustness & transfer.**
Does the inferred reward predict unseen trajectories, other individuals,
other gradient conditions?

## Progress

- [ ] Phase 0a — tabular simulation validation
- [ ] Phase 0b — continuous simulation validation
- [ ] Phase 1 — state/action representation
- [ ] Phase 2 — real data
- [ ] Phase 3 — comparison to the known mechanism
- [ ] Phase 4 — robustness & transfer

*(nothing is implemented yet — this README documents the starting point)*

## Technologies used

- Python
- NumPy / SciPy
- Gymnasium
- Stable-Baselines3 (PPO — the inner policy optimizer for Guided Cost
  Learning, and reused internally by `imitation`'s AIRL)
- `imitation` (AIRL/GAIL)
- Matplotlib
- JAX (optional, for differentiable components)
- `trackpy` (optional, if starting from raw video)
- TensorFlow, PyGame, `wandb` (optional — only needed for Phase 0b,
  pulled in transitively through the NanoGoal-RL submodule)

## Repository structure

```
Swim-IRL/
  README.md
  requirements.txt
  external/
    NanoGoal-RL/        # git submodule, pinned to v2 — continuous testbed (Phase 0b)
  sim/
    gridworld.py         # tabular simulated swimmer (Phase 0a), reused for Phase 2
    nanogoal_adapter.py  # thin wrapper exposing NanoGoal-RL's env as-is (Phase 0b)
  irl/
    maxent_linear.py     # Phase 0a
    gcl.py               # Guided Cost Learning / Deep MaxEnt (Phase 0b, Phase 1)
    airl_wrapper.py       # via imitation (Phase 0b, Phase 1)
  features.py             # isolable reward features
  data/
    loaders.py            # loading + cleaning of real trajectories
    simulate.py            # ground-truth trajectory generation (Phase 0a)
  eval/
    recovery.py            # reward recovery metrics (Phase 0a/0b)
    predictive.py           # out-of-sample prediction (Phase 4)
  experiments/             # reproducible scripts, one per milestone
  tests/                   # at least the Phase 0a/0b tests
```

## Installation

Tested to work identically on Windows, macOS, and Linux. Line endings are
normalized via `.gitattributes` to avoid CRLF/LF diff noise across
platforms.

```bash
git clone https://github.com/Josh012006/Swim-IRL.git
cd Swim-IRL
```

This repository uses [NanoGoal-RL v2](https://github.com/Josh012006/NanoGoal-RL/tree/v2)
(the `v2` branch) as a git submodule — it's the continuous
synthetic testbed for Phase 0b, not a runtime dependency of Phase 0a/1's
tabular work:

```bash
git submodule update --init --recursive
```

Create and activate a virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```bat
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Then, on any OS:
```bash
pip install -r requirements.txt
```

Phase 0b additionally needs NanoGoal-RL's own dependencies (TensorFlow,
Stable-Baselines3, PyGame, `wandb`) — install them only if you're working
on that phase:
```bash
pip install -r external/NanoGoal-RL/requirements.txt
```

## Usage

No runnable experiments yet — Phase 0a hasn't been implemented. This section
will be filled in with the exact reproducible commands (e.g.
`python experiments/phase0a_recovery.py --seed 0`) as soon as it lands, the
same way every subsequent milestone will get its own command here.

## Reproducibility

Every experiment must be rerunnable end-to-end from a single command with a
fixed seed. Results (figures, metrics) are regenerated by that command, not
hand-edited or committed as one-off artifacts.

## Limitations

**Known from the start (principles, not empirical findings yet):**
- IRL is ill-posed: several reward functions can explain the same behavior.
  This project will never claim to have found *the* worm's reward, only *a*
  simple reward consistent with what is observed.
- The inferred reward is an *as-if* model, not a claim about what the
  worm's nervous system actually computes.

**To be documented as the project progresses** (nothing yet — this section
will be updated at every milestone, including when a simplifying assumption
turns out to be a problem).

## Future work

- Cross-species comparison
- IRL on molecular motors or regulatory networks
- Link between the inferred reward and a signaling pathway
- Modular/transferable reward
- Bridge toward encoding a policy in a molecular substrate

## Author

Josué Mongan

## License

MIT License