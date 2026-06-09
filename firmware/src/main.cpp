#include <algorithm>
#include <cmath>

#include <mpc_generated_config.hpp>
#include "mpc_problem.hpp"
#include <tinympc/tiny_api.hpp>

#include <Arduino.h>

using Eigen::Matrix;
using Eigen::Dynamic;

namespace {

using VectorNx = Matrix<tinytype, Dynamic, 1>;

tinytype current_state[mpc_config::NX];
tinytype last_applied_torque[mpc_config::NU];
TinySolver* solver = nullptr;

tinytype square_ref_axis(tinytype t, tinytype phase, tinytype amp) {
    constexpr tinytype period = mpc_config::REFERENCE_PERIOD;
    tinytype shifted = std::fmod(t + phase * period, period);
    if (shifted < 0.0) {
        shifted += period;
    }
    return shifted < period / 2.0 ? amp : -amp;
}

void fill_single_standard_reference(tinytype start_t) {
    constexpr tinytype two_pi = static_cast<tinytype>(6.2831853071795864769);
    constexpr tinytype frequency = two_pi / mpc_config::REFERENCE_PERIOD;
    for (int k = 0; k < mpc_config::HORIZON; ++k) {
        const tinytype tk = start_t + static_cast<tinytype>(k) * mpc_config::DT;
        const tinytype phase = frequency * tk;
        const tinytype q1_ref = mpc_config::REFERENCE_START_SIGN *
            mpc_config::REFERENCE_AMPLITUDE_X * std::sin(phase);
        const tinytype q1_dot_ref = mpc_config::REFERENCE_START_SIGN *
            mpc_config::REFERENCE_AMPLITUDE_X * frequency * std::cos(phase);
        const tinytype q1_ddot_ref = -mpc_config::REFERENCE_START_SIGN *
            mpc_config::REFERENCE_AMPLITUDE_X * frequency * frequency * std::sin(phase);
        const tinytype torque_ref =
            mpc_config::GRAVITY_TORQUE_GAIN * q1_ref - mpc_config::PENDULUM_INERTIA * q1_ddot_ref;

        solver->work->Xref(0, k) = q1_ref / mpc_config::STATE_SCALE[0];
        solver->work->Xref(1, k) = q1_dot_ref / mpc_config::STATE_SCALE[1];
        solver->work->Xref(2, k) = 0.0;
        solver->work->Xref(3, k) = torque_ref / mpc_config::STATE_SCALE[3];
    }
}

void fill_reference(tinytype start_t) {
    if (mpc_config::MODEL_KIND == mpc_config::MODEL_SINGLE_STANDARD) {
        fill_single_standard_reference(start_t);
        return;
    }

    for (int k = 0; k < mpc_config::HORIZON; ++k) {
        const tinytype tk = start_t + static_cast<tinytype>(k) * mpc_config::DT;
        solver->work->Xref(0, k) =
            square_ref_axis(tk, 0.0, mpc_config::REFERENCE_AMPLITUDE_X) / mpc_config::STATE_SCALE[0];
        solver->work->Xref(1, k) =
            square_ref_axis(tk, 0.25, mpc_config::REFERENCE_AMPLITUDE_Y) / mpc_config::STATE_SCALE[1];
        for (int row = 2; row < mpc_config::NX; ++row) {
            solver->work->Xref(row, k) = 0.0;
        }
    }
}

void set_scaled_x0() {
    VectorNx x0(mpc_config::NX);
    for (int i = 0; i < mpc_config::NX; ++i) {
        x0(i) = current_state[i] / mpc_config::STATE_SCALE[i];
    }
    tiny_set_x0(solver, x0);
}

void clip_to_motor_envelope(const tinytype raw[mpc_config::NU], tinytype applied[mpc_config::NU]) {
    for (int axis = 0; axis < mpc_config::NU; ++axis) {
        const tinytype wheel = current_state[mpc_config::WHEEL_START + axis];
        const tinytype hi = std::min(
            mpc_config::STALL_TORQUE * (static_cast<tinytype>(1.0) - wheel / mpc_config::FREE_SPEED),
            mpc_config::SATURATION_TORQUE);
        const tinytype lo = std::max(
            -mpc_config::STALL_TORQUE * (static_cast<tinytype>(1.0) + wheel / mpc_config::FREE_SPEED),
            -mpc_config::SATURATION_TORQUE);
        applied[axis] = std::min(hi, std::max(lo, raw[axis]));
    }
}

void advance_linear_state(const tinytype applied[mpc_config::NU]) {
    tinytype input_scaled[mpc_config::NU];
    for (int axis = 0; axis < mpc_config::NU; ++axis) {
        const tinytype previous_torque = current_state[mpc_config::TORQUE_START + axis];
        const tinytype torque_rate = (applied[axis] - previous_torque) / mpc_config::DT;
        input_scaled[axis] = torque_rate / mpc_config::INPUT_SCALE[axis];
    }

    tinytype next_state[mpc_config::NX];
    for (int row = 0; row < mpc_config::NX; ++row) {
        tinytype value = 0.0;
        for (int col = 0; col < mpc_config::NX; ++col) {
            value += mpc_config::A_DYN[row][col] * (current_state[col] / mpc_config::STATE_SCALE[col]);
        }
        for (int col = 0; col < mpc_config::NU; ++col) {
            value += mpc_config::B_DYN[row][col] * input_scaled[col];
        }
        next_state[row] = value * mpc_config::STATE_SCALE[row];
    }

    for (int i = 0; i < mpc_config::NX; ++i) {
        current_state[i] = next_state[i];
    }
    for (int axis = 0; axis < mpc_config::NU; ++axis) {
        current_state[mpc_config::TORQUE_START + axis] = applied[axis];
        last_applied_torque[axis] = applied[axis];
    }
}

void reset_problem_state() {
    for (int i = 0; i < mpc_config::NX; ++i) {
        current_state[i] = mpc_config::INITIAL_STATE[i];
    }
    for (int i = 0; i < mpc_config::NU; ++i) {
        last_applied_torque[i] = current_state[mpc_config::TORQUE_START + i];
    }
}

void run_benchmark() {
    reset_problem_state();

    unsigned long min_us = 0xffffffffUL;
    unsigned long max_us = 0;
    unsigned long long sum_us = 0;
    int over_budget = 0;
    int solved_count = 0;

    Serial.printf(
        "MPC_BENCH_BEGIN,dt_us=%lu,steps=%d,horizon=%d\n",
        mpc_config::DT_US,
        mpc_config::BENCHMARK_STEPS,
        mpc_config::HORIZON);
    Serial.flush();

    for (int k = 0; k < mpc_config::BENCHMARK_STEPS; ++k) {
        const tinytype t = static_cast<tinytype>(k) * mpc_config::DT;
        set_scaled_x0();
        fill_reference(t);

        const unsigned long t0 = micros();
        const int exitflag = tiny_solve(solver);
        const unsigned long elapsed_us = micros() - t0;

        tinytype raw[mpc_config::NU];
        tinytype applied[mpc_config::NU];
        for (int axis = 0; axis < mpc_config::NU; ++axis) {
            const int torque_idx = mpc_config::TORQUE_START + axis;
            raw[axis] = solver->solution->x(torque_idx, 1) * mpc_config::STATE_SCALE[torque_idx];
        }
        clip_to_motor_envelope(raw, applied);
        advance_linear_state(applied);

        min_us = std::min(min_us, elapsed_us);
        max_us = std::max(max_us, elapsed_us);
        sum_us += elapsed_us;
        if (elapsed_us > mpc_config::DT_US) {
            over_budget += 1;
        }
        if (exitflag == 0 && solver->solution->solved) {
            solved_count += 1;
        }

        Serial.printf(
            "MPC_TIMING,k=%d,solve_us=%lu,dt_us=%lu,meets_dt=%d,exitflag=%d,solved=%d,iter=%d,raw_u0_nNm=%ld,applied_u0_nNm=%ld\n",
            k,
            elapsed_us,
            mpc_config::DT_US,
            elapsed_us <= mpc_config::DT_US ? 1 : 0,
            exitflag,
            solver->solution->solved,
            solver->solution->iter,
            static_cast<long>(std::lround(raw[0] * 1.0e9)),
            static_cast<long>(std::lround(applied[0] * 1.0e9)));
        Serial.flush();
        delay(1);
    }

    const unsigned long mean_us = static_cast<unsigned long>(
        sum_us / static_cast<unsigned long long>(mpc_config::BENCHMARK_STEPS));
    Serial.printf(
        "MPC_BENCH_SUMMARY,steps=%d,solved=%d,dt_us=%lu,min_us=%lu,mean_us=%lu,max_us=%lu,over_budget=%d,meets_dt=%d\n",
        mpc_config::BENCHMARK_STEPS,
        solved_count,
        mpc_config::DT_US,
        min_us,
        mean_us,
        max_us,
        over_budget,
        over_budget == 0 ? 1 : 0);
    Serial.flush();
}

}  // namespace

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.printf("MPC_BOOT\n");
    Serial.flush();
    Serial.printf("MPC_STAGE,setup_solver_begin\n");
    Serial.flush();
    const unsigned long setup_start_us = micros();
    solver = setup_mpc_solver();
    const unsigned long setup_us = micros() - setup_start_us;
    Serial.printf("MPC_SETUP,setup_us=%lu,ok=%d\n", setup_us, solver != nullptr ? 1 : 0);
    Serial.flush();
    if (solver == nullptr) {
        return;
    }
    Serial.printf("MPC_STAGE,benchmark_begin\n");
    Serial.flush();
    run_benchmark();
}

void loop() {
    delay(1000);
}
