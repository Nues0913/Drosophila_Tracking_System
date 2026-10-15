# Simulation and world adapter notes

This implementation was authored and reviewed by reading source only. No test,
training, dependency installation, build, import check, or runtime check was run
in this turn. The tests in `python/tests/test_sensing_world.py` are unexecuted.

## Shared sensing

`HexEye(contract)` exposes `directions_rad[2,N,2]` in robot-frame yaw/pitch
radians, left eye first. `compute_directions(contract)` returns the same table.
`HexEye.sha256()` hashes its little-endian float64 C-order bytes. The separate
sensor-contract hash identifies all settings and the geometry algorithm version.

`sample_image(image, eye=0)` normalizes integer grayscale/RGB input to `[0,1]`
and applies a cached Gaussian sparse pixel projector. `projection()` exposes
row indices, pixel indices and weights without a dense matrix. Acceptance angle
is the Gaussian standard deviation, truncated at three standard deviations.
`raycast(agent_pose,walls,disks,eye)` uses the same angular direction table and
Gaussian angular quadrature clipped to the same camera frustum. Directions are
obtained by back-projecting a hex lattice on the camera plane, so retinal corners
also lie within the pinhole field of view. It models extruded disks, 1 m walls and planar
motion. Default camera placement matches SDF: 0.14 m forward, ±0.06 m lateral,
0.27 m above floor; target height is 0.5 m. Fields of view come from the contract.

`ORNAdapter(num_types=1,K=.3,hill_n=2,adaptation_tau_s=2,K_floor=1e-4)` has
state `K[...,2,num_types]`. `reset(batch_shape=())` restores each type's baseline.
`update(concentration[...,2],dt,valid=None)` returns float32 `[...,2,num_types]`:
use the old K for `r=c**n/(c**n+K**n)`, then
`K=max(K_floor,K+min(dt/tau,1)*(c-K))`. Invalid/NaN readings output zero and
freeze K. The environment returns raw concentration; the policy owns ORN state.

`SensorCorruptor(seed)` has independent RNG streams for both cameras and odor.
Blackout is a valid black frame. Dropout skips transport acquisition; previously
received values remain available until the 200 ms contract watchdog expires.
Noise, brightness and dropout are configurable; unknown scenario settings fail.
2D noise is applied to sampled ommatidia, whereas ROS noise is applied to camera
pixels before projection. Their noise variance after projection therefore needs
measurement and calibration before claiming distribution-identical observations.

## World and Gymnasium contract

`TargetTrajectory(TrajectoryConfig|dict,seed).state(time_s)` returns
`pose[x,y,yaw],velocity[vx,vy]`. Motion choices are `stationary`, `linear` (wall
reflection) and `curved` (circle). Default arena is centered at (0,0), spanning
[-5,5] in both axes; targets remain within [-4.5,4.5].

`make_plume('puff'|'gaussian',PlumeConfig|dict,seed)` returns an object with
`reset(seed,time_s=0)`, `advance(time_s,sources)` and `concentration(points)`.
Puffs use a fixed 10 ms internal grid, advection, seeded turbulent diffusion,
spatial diffusion, decay and bounded storage. Sampling consumes no randomness.
Offline sources can be a time→positions function. ROS uses measured target
poses, so different source update schedules can still produce slight differences.
The Gaussian curriculum follows the current source and leaves no historical trail.

`FlyTrackEnv(contract=None,scenario='normal',**kwargs)` follows Gymnasium:
`reset(seed=...) -> (obs,info)` and
`step(action) -> (obs,reward,terminated,truncated,info)`. Obs contains only
float32 `vision[2,N]`, raw `odor[2]`, and `valid[3]` (left/right/odor).
The environment holds the latest samples on independent 30/50 Hz clocks.
Actions are physical `[v_m_s,omega_rad_s]`, with velocity and acceleration
limits from the contract. All sensors stale triggers an immediate zero command.
Collision terminates; duration truncates. Call `reset` after either ending.

Supported scenario names: `normal`, `occlusion`, `noise`, `conflict`. Occlusion
uses timed blackout in this first implementation. Conflict adds an odor decoy.
Common kwargs: `target_motion`, `target_speed_m_s`, `plume_model`, `plume_config`,
`trajectory_config`, `scenario_config` (strict dict loaded by the caller),
`episode_seconds`, `arena_size_m`, `arena_center_xy`, `initial_agent_pose`,
`obstacles` (extruded circle dicts), `domain_randomization`, `reward_weights`,
`subrays`, `render_mode='rgb_array'`. Unknown kwargs raise errors. Episode duration
overrides the scenario file's `duration_s` for both offline and ROS adapters.

`info['privileged_state']` is float32[12], ordered:

1. agent x, agent y (m)
2. sin(agent yaw), cos(agent yaw)
3. target x, target y (m)
4. target vx, target vy (m/s)
5. applied previous v (m/s), applied previous omega (rad/s)
6. wind x, wind y (m/s)

Other info includes teacher action, applied action, target bearing/distance,
visibility, occluded, collision, loss, time, episode/seed IDs, sensor timestamps,
and poses. None is part of actor observation. Done info adds `episode_metrics`
and `episode`; reacquisition is NaN if no event completed and has a censor flag.
The teacher slows for heading error and safe distance, with acceleration limits;
it does not perform obstacle route planning.

`make_vector_env(num_envs,contract,scenario='normal',seed=0,**kwargs)` uses
Gymnasium >=1.1 `SyncVectorEnv` with `AutoresetMode.SAME_STEP`. Ended episodes
reset before the returned next observation. Real terminal observation/info live
in `final_obs`/`final_info` and mask keys `_final_obs`/`_final_info`. Initial seeds
are seed+i; subsequent unseeded resets increment each environment's episode seed.
This is CPU vector orchestration, not a GPU-batched world implementation.

## ROS episode and topic protocol

All four nodes use simulation time supplied by bringup. Common parameters:
`sensor_contract_json`, `scenario`, `seed`, `episode_seconds`, `world_frame`,
`scenario_config_path` (JSON/YAML; PyYAML is confined to ROS adapters).

| Node | Inputs | Outputs |
| --- | --- | --- |
| odor_field | `/world/agent/pose`, `/world/target/pose`, `/world/scenario` | `/sim/odor` at contract odor_hz |
| target_mover | `/world/scenario`; `/world/tracking/set_pose` service | target SetEntityPose requests |
| scenario_manager | both world poses; SetEntityPose and policy reset service responses | `/world/scenario`; zero `/cmd_vel` while inactive |
| sensor_corruptor | `/sim/camera/{left,right}/image_raw`, `/sim/odor`, `/world/scenario` | `/fly/camera/{left,right}/image_raw`, `/fly/odor` |

Camera/odor QoS is best effort depth 2/5; world poses are reliable; scenario
events are reliable/transient-local depth 1. Header acquisition stamps are
preserved through corruption. Unsupported image encoding/row layout is rejected.

Reset protocol: publish inactive event → gate forwarded sensors → call Trigger
`/connectome_policy_node/reset` to clear cached commands → repeatedly send zero
velocity for at least `max(v_max/linear_accel_max,omega_max/angular_accel_max)+.1`
simulation seconds → SetEntityPose reset agent, then target → call policy Trigger
reset again to establish a fresh-sensor barrier → all successful responses →
active start event. The settling phase uses timers and does not block callbacks.
The active event carries the authoritative `episode_start_s`, seed, both initial
poses, trajectory/plume configs, arena and collision geometry. Receiving a new
active episode ID resets plume, trajectory and corruptor. Occlusion phase events
retain the same ID/epoch and preserve state. `active=false,phase='completed'`
ends a live episode. Clock rollback gates sensors and coordinates another reset.
`parameters_json.training_seed` stays fixed for seed-level statistics;
`environment_seed`/event.seed increments by episode. Override manager parameter
`training_seed` to match the trained artifact rather than its environment seed.
Forwarding rejects stamps at/before the new epoch to exclude queued old frames.
Service rejection/timeout keeps the episode inactive and retries without opening
the sensor gate. `max_episodes=0` runs until externally stopped.

Truth-based collision is a geometric proxy (circle robot, circle target, walls),
not Gazebo contact ground truth; the SDF robot uses a matching cylinder body plus wheels.
Configured ROS obstacles must also be physically created in SDF; this code does
not spawn geometry. Pausing sensor transport isolates policy state during reset,
but Gazebo wheel/joint velocities still require runtime confirmation after pose
teleport. Manager repeats zero commands during the configured inter-episode pause.

SetEntityPose request fields `entity` and `pose`, response `success`, and entity
MODEL constant were checked against upstream source, without running ROS:
[service definition](https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_interfaces/srv/SetEntityPose.srv),
[entity definition](https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_interfaces/msg/Entity.msg).

## Deferred verification

Run the authored behavior tests during the user-requested later validation phase. Then validate
projection orientation and photometric statistics against Gazebo camera frames;
compare deterministic source traces at identical update schedules; verify
Gymnasium terminal data and curriculum resets; build ROS messages/packages and
exercise policy truth-topic exclusion, resets, 200 ms watchdog, clock rollback,
header preservation, QoS discovery, SetEntityPose bridging and complete episodes.
Measure latency on the actual GPU host and calibrate collision proxies against
Gazebo contacts. No empirical success or timing claim is made here.
