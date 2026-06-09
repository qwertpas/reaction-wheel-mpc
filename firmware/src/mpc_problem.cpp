#include "mpc_problem.hpp"

#include <mpc_generated_config.hpp>

#include <Arduino.h>

namespace {

TinySolver* solver = nullptr;

tinyMatrix make_state_bounds(bool upper) {
    tinyMatrix bounds(mpc_config::NX, mpc_config::HORIZON);
    for (int k = 0; k < mpc_config::HORIZON; ++k) {
        for (int row = 0; row < mpc_config::NX; ++row) {
            tinytype value = upper ? 1.0e9 : -1.0e9;
            if (row >= mpc_config::TORQUE_START && row < mpc_config::TORQUE_START + mpc_config::NU) {
                value = upper ? mpc_config::SATURATION_TORQUE : -mpc_config::SATURATION_TORQUE;
            }
            bounds(row, k) = value / mpc_config::STATE_SCALE[row];
        }
    }
    return bounds;
}

tinyMatrix make_input_bounds(bool upper) {
    tinyMatrix bounds(mpc_config::NU, mpc_config::HORIZON - 1);
    for (int k = 0; k < mpc_config::HORIZON - 1; ++k) {
        for (int row = 0; row < mpc_config::NU; ++row) {
            const tinytype value = upper ? mpc_config::INPUT_BOUND_LIMIT : -mpc_config::INPUT_BOUND_LIMIT;
            bounds(row, k) = value / mpc_config::INPUT_SCALE[row];
        }
    }
    return bounds;
}

tinyMatrix make_A() {
    tinyMatrix A(mpc_config::NX, mpc_config::NX);
    for (int row = 0; row < mpc_config::NX; ++row) {
        for (int col = 0; col < mpc_config::NX; ++col) {
            A(row, col) = mpc_config::A_DYN[row][col];
        }
    }
    return A;
}

tinyMatrix make_B() {
    tinyMatrix B(mpc_config::NX, mpc_config::NU);
    for (int row = 0; row < mpc_config::NX; ++row) {
        for (int col = 0; col < mpc_config::NU; ++col) {
            B(row, col) = mpc_config::B_DYN[row][col];
        }
    }
    return B;
}

tinyMatrix make_Q() {
    tinyMatrix Q = tinyMatrix::Zero(mpc_config::NX, mpc_config::NX);
    for (int i = 0; i < mpc_config::NX; ++i) {
        Q(i, i) = mpc_config::Q_DIAG[i];
    }
    return Q;
}

tinyMatrix make_R() {
    tinyMatrix R = tinyMatrix::Zero(mpc_config::NU, mpc_config::NU);
    for (int i = 0; i < mpc_config::NU; ++i) {
        R(i, i) = mpc_config::R_DIAG[i];
    }
    return R;
}

}  // namespace

TinySolver* setup_mpc_solver() {
    if (solver != nullptr) {
        return solver;
    }

    Serial.printf("MPC_STAGE,tiny_setup_begin\n");
    Serial.flush();
    const int status = tiny_setup(
        &solver,
        make_A(),
        make_B(),
        make_Q(),
        make_R(),
        mpc_config::RHO,
        mpc_config::NX,
        mpc_config::NU,
        mpc_config::HORIZON,
        make_state_bounds(false),
        make_state_bounds(true),
        make_input_bounds(false),
        make_input_bounds(true),
        0);
    Serial.printf("MPC_STAGE,tiny_setup_done,status=%d\n", status);
    Serial.flush();

    if (status != 0) {
        solver = nullptr;
        return nullptr;
    }

    Serial.printf("MPC_STAGE,tiny_settings_begin\n");
    Serial.flush();
    tiny_update_settings(
        solver->settings,
        mpc_config::ABS_PRI_TOL,
        mpc_config::ABS_DUA_TOL,
        mpc_config::MAX_ITER,
        mpc_config::CHECK_TERMINATION,
        1,
        1);
    Serial.printf("MPC_STAGE,tiny_settings_done\n");
    Serial.flush();

    return solver;
}
