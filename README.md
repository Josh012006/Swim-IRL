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
  testable one at a time.
- Planned extensions, in order: **Deep MaxEnt IRL**, then **AIRL**
  (Fu, Luo & Levine, 2018) via the `imitation` library (HumanCompatibleAI),
  for a more transferable reward over continuous state.
- Core difficulty: the worm's decisions are not observed directly, only its
  position/posture over time → heading, speed, and reorientation events
  (pirouette vs. weathervane) will need to be reconstructed from real
  trajectories before IRL can be run (Phase 2).

## Roadmap (phases)

**Phase 0 — Simulation validation (blocking).**
Build a simulated swimmer with a CHOSEN ground-truth reward, verify that
MaxEnt IRL recovers it, on ≥ 3 different test rewards. Real data is off
limits until this milestone is met.

**Phase 1 — State/action representation + features.**
Candidate state: local concentration, estimated local gradient, temporal
derivative of concentration, speed, heading. Candidate features:
concentration, velocity-gradient alignment, temporal derivative
(chemotaxis hypothesis), effort cost, information term (infotaxis,
Vergassola 2007).
**Open decision**: a unified action space (a single turning angle of
variable magnitude) vs. two explicit channels (continuous for weathervane +
discrete for pirouette, closer to the biology)? To be settled after reading
Ziebart 2008, before writing the simulator.

**Phase 2 — Real data.**
Load *C. elegans* trajectories in a gradient (NaCl or odorant). Cleaning,
interpolation, speed estimation, pirouette/weathervane segmentation.
Not allowed: passive (purely Brownian) particles → degenerate reward.

**Phase 3 — Comparison to the known mechanism.**
Does the "temporal derivative" feature emerge as dominant, consistent with
what is known about the real sensory mechanism?

**Phase 4 — Robustness & transfer.**
Does the inferred reward predict unseen trajectories, other individuals,
other gradient conditions?

## Progress

- [ ] Phase 0 — simulation validation
- [ ] Phase 1 — state/action representation
- [ ] Phase 2 — real data
- [ ] Phase 3 — comparison to the known mechanism
- [ ] Phase 4 — robustness & transfer

*(nothing is implemented yet — this README documents the starting point)*

## Technologies used

- Python
- NumPy / SciPy
- Gymnasium
- `imitation` (AIRL/GAIL)
- Matplotlib
- JAX (optional, for differentiable components)
- `trackpy` (optional, if starting from raw video)

## Repository structure

```
Swim-IRL/
  README.md
  requirements.txt
  sim/              # simulated swimmer (Phase 0), reused for Phase 2
  irl/
    maxent_linear.py
    deep_maxent.py
    airl_wrapper.py   # via imitation
  features.py         # isolable reward features
  data/
    loaders.py        # loading + cleaning of real trajectories
    simulate.py        # ground-truth trajectory generation (Phase 0)
  eval/
    recovery.py        # reward recovery metrics (Phase 0)
    predictive.py       # out-of-sample prediction (Phase 4)
  experiments/         # reproducible scripts, one per milestone
  tests/               # at least the Phase 0 tests
```

## Installation

Tested to work identically on Windows, macOS, and Linux. Line endings are
normalized via `.gitattributes` to avoid CRLF/LF diff noise across
platforms.

```bash
git clone https://github.com/Josh012006/Swim-IRL.git
cd Swim-IRL
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

## Usage

No runnable experiments yet — Phase 0 hasn't been implemented. This section
will be filled in with the exact reproducible commands (e.g.
`python experiments/phase0_recovery.py --seed 0`) as soon as it lands, the
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

Cross-species comparison; IRL on molecular motors or regulatory networks;
link between the inferred reward and a signaling pathway; modular/
transferable reward; bridge toward encoding a policy in a molecular
substrate.

## Author

Josué Mongan 

## License

MIT License