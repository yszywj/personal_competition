# Joint RL Core

`joint_rl_core` is a simulator-neutral foundation for a joint reinforcement-
learning controller.  It intentionally has no imports from `competition_envs`,
the old R9 code, Torch, or Gymnasium.  The concrete simulator command bridge is
outside this package.

## Lifecycle

Each controllable entity has an independent lifecycle:

```text
STAGED -- wait --> STAGED
STAGED -- activate(location, objective, movement) --> PENDING
PENDING -- accepted --> ACTIVE
PENDING -- rejected --> STAGED
ACTIVE -- explicit terminal evidence --> TERMINAL
```

Activation is one policy decision that carries a normalized location, an
initial objective slot, and the first movement command.  `PENDING` is an execution acknowledgement boundary,
not a separate policy stage.  A synchronous adapter may call
`confirm_activations()` immediately after it executes the intent.  Missing an
observation never marks an entity terminal by itself.  Every intent and receipt
contains the originating step, so a delayed receipt cannot confirm a newer
request for the same slot.  Explicit timeout helpers release requests whose
receipt never arrives.  Actions, receipts, timeouts, and explicit terminal
events share one non-decreasing step timeline; actions themselves remain
strictly increasing.

The current policy samples an accepted activation autoregressively as
`objective -> placement/first movement` after the activation gate.  Placement
and first-movement distributions are conditioned on the selected objective
embedding.  For `ACTIVE` reassignment, the movement distribution in that same
step is conditioned on the newly selected objective.

## Conditional action

Every joint decision contains one `UnitAction` for every stable entity slot.
The branches that affect probability, entropy, and the PPO ratio depend on the
entity state and sampled parent action:

| Entity state and choice | Active policy terms |
| --- | --- |
| `STAGED`, wait | activation |
| `STAGED`, activate | activation, placement, objective, movement |
| `PENDING` | none |
| `ACTIVE`, keep objective | retarget, movement |
| `ACTIVE`, update objective | retarget, objective, movement |
| `TERMINAL` | none |

Inactive values are canonicalized to stable no-op values.  The trajectory
record validates the exact set of active log-probability terms so a learner
cannot accidentally train on ignored parameters.

The shared sensor is represented as a team resource rather than as an
independent simulated vehicle.  `SharedSensorAction.requester_slots` supports a
configurable number of requests per step.  Eligibility, total capacity, and a
global cooldown are included in the mask.  Capacity is reserved while a request
is pending and consumed only after an accepted receipt; a rejected request does
not start cooldown.  A request remains reserved if its requester terminates,
because the external service may already have accepted it; only its matching
receipt or timeout releases the reservation.  When the resource head has at
least one eligible entity, its complete STOP/selection log probability is
stored under the `shared_sensor` term, including when STOP is selected.
Multi-request samples retain every without-replacement mask in
`SharedSensorPolicyTrace`.

The simulator-neutral core does not assume where that resource is implemented.
The concrete adapter distinguishes two backends.  The legacy `per_unit` backend
keeps a counter and active window on each requester.  The current
`team_global` backend keeps one counter and one active window on
`SimulatorFactory`; while that window is active, every requester is masked and
the effective limit is one accepted request per step.  A team-global request
may use a `STAGED` or `ACTIVE` slot as its command executor, so it can be issued
in the same simulator step as that slot's deployment and launch.  The legacy
per-unit backend only permits `ACTIVE` requesters.  Receipts are matched to the
corresponding per-unit or factory-level counter.  The backend type and its
capacity are part of the environment checkpoint contract.

## Observation

`JointObservationEncoder` produces one fixed-width NumPy vector per entity.  It
combines only caller-supplied public data with controller-owned lifecycle and
action history.  External IDs are metadata and never numeric input features.

Unknown positions and velocities use zero-valued payload fields together with
separate validity bits; known zeros therefore remain distinguishable from
missing data.  Objective slots carry relative position, distance, bearing,
velocity, age, a type one-hot, and four controller-owned assignment loads
(`total/high/medium/low`) when those fields are supplied.  Only `ACTIVE`
friendly units contribute to the loads; they contain no enemy health or
ground-truth state.  Each unit also carries the normalized reference distance
and progress fraction for its current distance potential, so the value
function observes the state on which motion shaping depends.  Before an
entity is deployed, target geometry is measured from the map centre while the
separate self-position validity bit remains false.  The placement head can
therefore see the map layout without receiving a fabricated deployment point.
All values are clipped to finite ranges.

With three unit types, three objective types, and `O` objective slots, the
encoder dimension is `40 + 19*O`.  When the CLI omits `--objective-slots`, the
adapter uses `max(18, len(reward_policy.objective_ids))`; an explicit value that
cannot hold all scored objectives is rejected.  Legacy/final24 use `O=18` and
382 dimensions per unit.  Final20 easy/medium/hard use `O=24/36/48` and
496/724/952 dimensions.  Checkpoints with different `O` or observation width
are incompatible, as is the previous 308-dimensional schema.

`StableSlotRegistry` maps integer or string IDs to fixed slots.  It sorts all
new IDs received in one frame before assignment, never exposes the IDs as
numeric features, never reuses a slot during an episode, and reports unique
overflow IDs.

`JointSpaceSpec` is shared by lifecycle tracking, masks, observation encoding,
and trajectory storage.  This prevents an action from selecting an objective
slot that is absent from the encoded observation.

`JointPolicyTrace` stores separate per-unit plan and motion values plus a team
sensor value.  `JointTransition` likewise carries `plan_rewards`,
`motion_rewards`, and `sensor_reward`, in addition to separate per-unit and team
`terminated`/`truncated` flags.  Time-limit truncation therefore remains
distinct from a terminal game outcome for each critic.  Planning credit may be
filled after the complete episode is available; motion and sensor rewards are
recorded at each environment step.  Per-entity sequences must already be
ordered by stable slot.  The buffer copies all masks, traces, and observations
into byte-backed read-only arrays, preventing later caller mutations from
changing a PPO sample.

The current PPO consumer forms three independent joint ratios and advantages:

- plan: activation, placement, objective, and retarget terms use a direct,
  undiscounted episode outcome;
- motion: movement terms use step rewards and per-unit GAE;
- sensor: the shared pointer/STOP term uses a team sensor reward and GAE.  On
  `team_global`, its potential is the age-decayed fraction of unique, legally
  observed interceptor tracks, normalized by the scenario's initial enemy
  interceptor count.  On legacy `per_unit`, it retains the legally discovered
  scored-objective potential so old runs preserve their reward semantics.

Plan samples are weighted inversely by the number of planning decisions made
by that unit in that episode, then normalized over the rollout.  This weight is
used for the plan policy loss, entropy, and advantage normalization, so a long
sequence of wait/keep decisions does not dominate the planning update.

## Integration boundary

`JointControlTracker.apply()` returns `ResolvedIntents`.  These are generic
activation, reassignment, movement, and shared-sensor intents.  No class in this
package translates them to an external simulator command.

The concrete `JointGameEnv` adapter performs the following transaction:

1. Encode the public frame and build masks.
2. Sample one complete `JointAction` and retain its masks and active terms.
3. Validate and resolve it to generic intents.
4. Submit intents through an application-specific adapter.
5. Feed explicit activation and shared-resource receipts back to the tracker.
6. Record motion and sensor rewards with the complete joint transition,
   including wait decisions, exact sampled masks, and separate
   terminal/truncation flags.
7. At episode end, attach each eligible unit's plan return.  Assignment
   responsibility uses the state snapshot taken before team-wide terminal
   conversion, preserving the final step.  The raw local objective credit
   remains conserved per target.  Its learning copy is
   `clip(eligible_count * raw_credit, 0, 1)` before the default
   `0.7 team / 0.3 local` mix, so the two terms have comparable per-unit scale.

The policy checkpoint schema remains v3.  The legacy `per_unit` environment
contract remains v3 for reproducibility.  The current `team_global` environment
contract is v4 and records `backend_capacity_team`, `backend_active_minutes`,
`effective_max_requests_per_step`, and the scenario-specific initial enemy
interceptor count used to normalize `detected_threat_count`, while retaining
the same policy tensor schema.  Resume and warm start reject a backend mismatch,
and they also reject different unit/objective spaces or observation widths.

Run the standalone tests from the parent directory:

```bash
cd /home/amax/ry/competition
/home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
  -m unittest -v personal_train.test_joint_rl_core
```
