
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
import tinympc
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--wheel-speed", type=float, default=0.0)
parser.add_argument("--period", type=float, default=5.0)
args = parser.parse_args()


body_mass = 0.100
body_height = 0.150
wheel_mass = 0.020
wheel_radius = 0.030
gravity = 9.81

M = body_mass + wheel_mass
C = body_height / 2.0
body_inertia = 1/12. * body_mass * body_height**2
wheel_inertia = 1/2. * wheel_mass * wheel_radius**2
pendulum_inertia = M * C**2 + body_inertia
coupled_inertia = pendulum_inertia + wheel_inertia
gravity_torque_gain = M * C * gravity
G = gravity_torque_gain / pendulum_inertia

dt = 0.02
duration = 20.0
reference_amplitude = 0.15
reference_frequency = 2.0 * np.pi / args.period

HORIZON = 30

motor_kv = 4300.0 * 2.0 * np.pi / 60.0
battery_voltage = 5.0
phase_resistance = 0.470
motor_KT = 1.0 / motor_kv
stall_torque = battery_voltage / phase_resistance * motor_KT
free_speed = battery_voltage * motor_kv
max_torque = 0.0150
torque_rate_scale = max_torque / dt



# gain matrices

mpc_state_cost = np.diag([1000.0, 10.0, 1.0e-6, 1.0])
mpc_torque_rate_cost = np.diag([1.0])
lqr_state_cost = np.diag([1000.0, 10.0, 1.0e-6]) #same as MPC
lqr_torque_cost = np.diag([1.0])


# LQR
lqr_continuous_a = np.array([
    [0.0, 1.0, 0.0],
    [G, 0.0, 0.0],
    [-G, 0.0, 0.0],
])
lqr_continuous_b = np.array([
    [0.0], 
    [-1/pendulum_inertia], 
    [coupled_inertia/(wheel_inertia*pendulum_inertia)]
])
lqr_a, lqr_b, _, _, _ = cont2discrete(
    (lqr_continuous_a, lqr_continuous_b, np.eye(3), np.zeros((3, 1))),
    dt,
)
lqr_cost_to_go = solve_discrete_are(lqr_a, lqr_b, lqr_state_cost, lqr_torque_cost)
lqr_gain = np.linalg.solve(
    lqr_b.T @ lqr_cost_to_go @ lqr_b + lqr_torque_cost,
    lqr_b.T @ lqr_cost_to_go @ lqr_a,
)



# MPC
mpc_continuous_a = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [G, 0.0, 0.0, -1/pendulum_inertia],
    [-G, 0.0, 0.0, coupled_inertia/(wheel_inertia*pendulum_inertia)],
    [0.0, 0.0, 0.0, 0.0],
])
mpc_continuous_b = np.array([
    [0.0],
    [0.0],
    [0.0],
    [1.0]
])
mpc_a, mpc_b, _, _, _ = cont2discrete((mpc_continuous_a, mpc_continuous_b, np.eye(4), np.zeros((4, 1))), dt)

state_scale = np.array([1.0, 1.0, free_speed, max_torque]) #speed and torque are very large/small, causes mpc solve to be unstable?
input_scale = np.array([torque_rate_scale])
scale_state = np.diag(1.0 / state_scale)
unscale_state = np.diag(state_scale)
scale_input = np.diag(input_scale)

mpc = tinympc.TinyMPC()
mpc.setup(
    scale_state @ mpc_a @ unscale_state,
    scale_state @ mpc_b @ scale_input,
    unscale_state.T @ mpc_state_cost @ unscale_state,
    scale_input.T @ mpc_torque_rate_cost @ scale_input,
    HORIZON,
    rho=1.0, #0.1
    max_iter=500,
    abs_pri_tol=1.0e-5,
    abs_dua_tol=1.0e-5,
    check_termination=5,
)
mpc.set_bound_constraints( #use big values, don't really care
    np.array([-1e9, -1e9, -1e9, -max_torque]) / state_scale,
    np.array([1e9, 1e9, 1e9, max_torque]) / state_scale,
    np.array([-1e9]) / input_scale,
    np.array([1e9]) / input_scale,
)
mpc.set_linear_constraints(
    np.array([
        [0.0, 0.0, stall_torque/free_speed, 1.0],
        [0.0, 0.0, -stall_torque/free_speed, -1.0],
    ]) @ unscale_state,
    stall_torque * np.ones(2),
    np.zeros((0,1)),
    np.zeros(0),
)
mpc.set_u_ref(np.zeros((1, HORIZON - 1)))




# integration
def rk4_step(state, torque):
    def dynamics(x):
        angle, rate, last_torque = x
        gravity_torque = gravity_torque_gain * np.sin(angle)
        return np.array([
            rate,
            (gravity_torque - torque) / pendulum_inertia,
            -gravity_torque / pendulum_inertia + coupled_inertia * torque / (wheel_inertia * pendulum_inertia),
        ])

    k1 = dynamics(state)
    k2 = dynamics(state + 0.5 * dt * k1)
    k3 = dynamics(state + 0.5 * dt * k2)
    k4 = dynamics(state + dt * k3)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


step_count = int(round(duration / dt))
time = np.arange(step_count + 1) * dt
trajectories = {}
for controller_name in ("LQR", "MPC"):
    body_angle_history = np.zeros(step_count + 1)
    body_rate_history = np.zeros(step_count + 1)
    wheel_speed_history = np.zeros(step_count + 1)
    reference_angle_history = np.zeros(step_count + 1)
    clipped_torque_history = np.zeros(step_count)

    state = np.array([0.0, 0.0, args.wheel_speed])
    clipped_torque = 0.0

    for i, current_time in enumerate(time[:-1]):
        body_angle, body_rate, wheel_speed = state
        body_angle_history[i] = body_angle
        body_rate_history[i] = body_rate
        wheel_speed_history[i] = wheel_speed

        phase = reference_frequency * current_time
        reference_angle = reference_amplitude * np.sin(phase)
        reference_rate = reference_amplitude*reference_frequency * np.cos(phase)
        reference_acceleration = -reference_amplitude*reference_frequency**2 * np.sin(phase)
        reference_torque = gravity_torque_gain*reference_angle - pendulum_inertia*reference_acceleration
        reference_angle_history[i] = reference_angle

        if controller_name == "LQR":
            reference_state = np.array([reference_angle, reference_rate, 0.0])
            requested_torque = reference_torque - (lqr_gain @ (state - reference_state)).item()
        else:
            preview_times = current_time + np.arange(HORIZON)*dt
            preview_phases = reference_frequency * preview_times
            preview_angles = reference_amplitude * np.sin(preview_phases)
            preview_rates = reference_amplitude*reference_frequency * np.cos(preview_phases)
            preview_accels = -reference_amplitude*reference_frequency**2 * np.sin(preview_phases)
            preview_array = np.zeros((4, HORIZON))
            preview_array[0] = preview_angles
            preview_array[1] = preview_rates
            preview_array[3] = gravity_torque_gain*preview_angles - pendulum_inertia*preview_accels #reference torque

            mpc_start_state = np.append(state, clipped_torque)
            mpc.set_x0(mpc_start_state / state_scale)
            mpc.set_x_ref(preview_array / state_scale[:, None])
            solution = mpc.solve()
            requested_torque = np.asarray(solution["states_all"])[1, 3] * state_scale[3]

        allowed_min_torque = max(-stall_torque * (1.0 + wheel_speed / free_speed), -max_torque)
        allowed_max_torque = min(stall_torque * (1.0 - wheel_speed / free_speed), max_torque)
        clipped_torque = np.clip(requested_torque, allowed_min_torque, allowed_max_torque)
        clipped_torque_history[i] = clipped_torque

        state = rk4_step(state, clipped_torque)

    body_angle_history[-1], body_rate_history[-1], wheel_speed_history[-1] = state
    reference_angle_history[-1] = reference_amplitude * np.sin(reference_frequency * time[-1])
    trajectories[controller_name] = {
        "body_angle": body_angle_history,
        "body_rate": body_rate_history,
        "wheel_speed": wheel_speed_history,
        "reference_angle": reference_angle_history,
        "applied_torque": clipped_torque_history,
    }

lqr = trajectories["LQR"]
mpc_run = trajectories["MPC"]
lqr_error = lqr["body_angle"] - lqr["reference_angle"]
mpc_error = mpc_run["body_angle"] - mpc_run["reference_angle"]
print(f"LQR tracking RMSE: {np.sqrt(np.mean(lqr_error**2)):.6f} rad")
print(f"MPC tracking RMSE: {np.sqrt(np.mean(mpc_error**2)):.6f} rad")





# trajecotry plots
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
axes[0].plot(time, lqr["reference_angle"], color="0.45", alpha=0.65, label="reference")
axes[0].plot(time, lqr["body_angle"], color="tab:blue", label="LQR")
axes[0].plot(time, mpc_run["body_angle"], color="tab:orange", label="MPC")
axes[0].set_ylabel("body angle [rad]")
axes[0].legend(ncol=3, fontsize=8, loc="upper right")

axes[1].plot(time[:-1], lqr["applied_torque"], color="tab:blue", label="LQR applied")
axes[1].plot(time[:-1], mpc_run["applied_torque"], color="tab:orange", label="MPC applied")
axes[1].set_ylabel("torque [N m]")
axes[1].set_ylim(-1.1 * max_torque, 1.1 * max_torque)

axes[2].plot(time, lqr["wheel_speed"], color="tab:blue", label="LQR")
axes[2].plot(time, mpc_run["wheel_speed"], color="tab:orange", label="MPC")
axes[2].set_ylabel("wheel speed [rad/s]")
axes[2].set_xlabel("time [s]")
axes[2].set_ylim(-1.1 * free_speed, 1.1 * free_speed)

angle_values = np.concatenate((lqr["reference_angle"], lqr["body_angle"], mpc_run["body_angle"]))
angle_min = np.min(angle_values)
angle_max = np.max(angle_values)
angle_margin = max(0.08 * (angle_max - angle_min), 1.0e-3)
axes[0].set_ylim(angle_min - angle_margin, angle_max + angle_margin)

for axs in axes:
    axs.grid(True, alpha=0.35)

fig.suptitle(f"Reference period {args.period:.1f} s, initial wheel speed {args.wheel_speed:.0f} rad/s")
fig.tight_layout()
fig.savefig("trajectories.png", dpi=180)



#torque speed plots
fig, axs = plt.subplots(figsize=(8, 5))
speed_grid = np.linspace(-free_speed, free_speed, 700)
torque_min_grid = np.maximum(-stall_torque * (1.0 + speed_grid / free_speed), -max_torque)
torque_max_grid = np.minimum(stall_torque * (1.0 - speed_grid / free_speed), max_torque)
axs.fill_between(speed_grid, torque_min_grid, torque_max_grid, color="0.65", alpha=0.25, label="allowed region")
axs.plot(speed_grid, torque_min_grid, color="0.25", linewidth=1.0)
axs.plot(speed_grid, torque_max_grid, color="0.25", linewidth=1.0)
axs.axhline(max_torque, color="0.45", linestyle=":", linewidth=1.0)
axs.axhline(-max_torque, color="0.45", linestyle=":", linewidth=1.0)
axs.axhline(0, color="k", linewidth=0.5)
axs.axvline(0, color="k", linewidth=0.5)
axs.axvline(free_speed, color="0.45", linestyle=":", linewidth=1.0)
axs.axvline(-free_speed, color="0.45", linestyle=":", linewidth=1.0)
axs.plot(lqr["wheel_speed"][:-1], lqr["applied_torque"], color="tab:blue", linewidth=2)
axs.plot(mpc_run["wheel_speed"][:-1], mpc_run["applied_torque"], color="tab:orange", linewidth=2)
axs.scatter(lqr["wheel_speed"][0], lqr["applied_torque"][0], color="tab:blue", label="LQR")
axs.scatter(mpc_run["wheel_speed"][0], mpc_run["applied_torque"][0], color="tab:orange", label="MPC")
axs.set_xlabel("Wheel speed [rad/s]")
axs.set_ylabel("Motor torque [Nm]")
axs.set_title("Motor Torque-Speed Envelope")
axs.set_xlim(-1.1 * free_speed, 1.1 * free_speed)
axs.set_ylim(-1.1 * max_torque, 1.1 * max_torque)
axs.grid(True, alpha=0.35)
axs.legend(fontsize=8)
fig.tight_layout()
fig.savefig("torquespeed.png", dpi=180)

