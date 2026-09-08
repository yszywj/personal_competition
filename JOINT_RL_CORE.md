# Joint RL Core

`joint_rl_core` is a simulator-neutral foundation for a joint reinforcement-
learning controller.  It intentionally has no imports from `competition_envs`,
the old R9 code, Torch, or Gymnasium.  The concrete simulator command bridge is
outside this package.

## Lifecycle

Each controllable entity has an independent lifecycle:

```text
STAGED -- wait --> STAGED
STAGED -- activate(location, objective) --> PENDING
PENDING -- accepted --> ACTIVE
PENDING -- rejected --> STAGED
ACTIVE -- explicit terminal evidence --> TERMINAL
```

Activation is one policy decision that carries a normalized location and an
initial objective slot.  `PENDING` is an execution acknowledgement boundary,
not a separate policy stage.  A synchronous adapter may call
`confirm_activations()` immediately after it executes the intent.  Missing an
observation never marks an entity terminal by itself.  Every intent and receipt
contains the originating step, so a delayed receipt cannot confirm a newer
request for the same slot.  Explicit timeout helpers release requests whose
receipt never arrives.  Actions, receipts, timeouts, and explicit terminal
events share one non-decreasing step timeline; actions themselves remain
strictly increasing.

## Conditional action

Every joint decision contains one `UnitAction` for every stable entity slot.
The branches that affect probability, entropy, and the PPO ratio depend on the
entity state and sampled parent action:

| Entity state and choice | Active policy terms |
| --- | --- |
| `STAGED`, wait | activation |
| `STAGED`, activate | activation, placement, objective |
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

## Observation

`JointObservationEncoder` produces one fixed-width NumPy vector per entity.  It
combines only caller-supplied public data with controller-owned lifecycle and
action history.  External IDs are metadata and never numeric input features.

Unknown positions and velocities use zero-valued payload fields together with
separate validity bits; known zeros therefore remain distinguishable from
missing data.  Objective slots carry relative position, distance, bearing,
velocity, age, and a type one-hot when those fields are supplied.  All values
are clipped to finite ranges.

`StableSlotRegistry` maps integer or string IDs to fixed slots.  It sorts all
new IDs received in one frame before assignment, never exposes the IDs as
numeric features, never reuses a slot during an episode, and reports unique
overflow IDs.

`JointSpaceSpec` is shared by lifecycle tracking, masks, observation encoding,
and trajectory storage.  This prevents an action from selecting an objective
slot that is absent from the encoded observation.

`JointTransition` stores a value, reward, `terminated`, and `truncated` flag for
every entity.  It also has an explicit team value/reward/termination channel for
the shared-resource factor.  Time-limit truncation therefore remains distinct
from a terminal game outcome.  Per-entity sequences must already be ordered by
stable slot.  The buffer copies all masks, traces, and observations into
byte-backed read-only arrays, preventing later caller mutations from changing a
PPO sample.

## Integration boundary

`JointControlTracker.apply()` returns `ResolvedIntents`.  These are generic
activation, reassignment, movement, and shared-sensor intents.  No class in this
package translates them to an external simulator command.

A future benign game adapter should perform the following transaction:

1. Encode the public frame and build masks.
2. Sample one complete `JointAction` and retain its masks and active terms.
3. Validate and resolve it to generic intents.
4. Submit intents through an application-specific adapter.
5. Feed explicit activation and shared-resource receipts back to the tracker.
6. Record the complete joint transition, including wait decisions, the exact
   sampled masks, and separate terminal/truncation flags.

Run the standalone tests from the parent directory:

```bash
cd /home/amax/ry/competition
python3 -m unittest -v personal_train.test_joint_rl_core
```
