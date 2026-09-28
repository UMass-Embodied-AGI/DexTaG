#include "xarm_kinematics_interface.h"

// Thin extern "C" wrapper to resolve C++ overloads for ctypes access.
// All arrays are flat (row-major for 4x4 matrices).
// Batch functions: FK uses OpenMP; IK must be sequential (see note below).

extern "C" {

// ---- Single-element functions ----

int xarm7_config_c(double *q_max, double *q_min, double *tcp_offset, double *world_offset) {
    return xarm7_config(q_max, q_min, tcp_offset, world_offset);
}

int xarm7_fk_mat_c(double *theta, double *tcp_mat) {
    return xarm7_forward_kinematics(theta, reinterpret_cast<double(*)[4]>(tcp_mat));
}

int xarm7_fk_pose_c(double *theta, double *pose_rpy) {
    return xarm7_forward_kinematics(theta, pose_rpy);
}

int xarm7_ik_mat_c(double *tcp_mat, double *q_pre, double *theta) {
    return xarm7_inverse_kinematics(reinterpret_cast<double(*)[4]>(tcp_mat), q_pre, theta);
}

int xarm7_ik_pose_c(double *pose_rpy, double *q_pre, double *theta) {
    return xarm7_inverse_kinematics(pose_rpy, q_pre, theta);
}

// ---- Batch functions ----
// ret_codes[i] holds the per-element return code (0 = success).
//
// NOTE: xarm7_inverse_kinematics uses internal mutable state, so IK
// batch must run sequentially to produce deterministic, order-consistent
// results.  FK is stateless and safe to parallelize with OpenMP.

int xarm7_ik_mat_batch_c(double *tcp_mats, double *q_refs, double *thetas,
                         int *ret_codes, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        ret_codes[i] = xarm7_inverse_kinematics(
            reinterpret_cast<double(*)[4]>(tcp_mats + i * 16),
            q_refs + i * 7,
            thetas + i * 7);
    }
    return 0;
}

int xarm7_ik_pose_batch_c(double *poses, double *q_refs, double *thetas,
                          int *ret_codes, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        ret_codes[i] = xarm7_inverse_kinematics(
            poses + i * 6,
            q_refs + i * 7,
            thetas + i * 7);
    }
    return 0;
}

int xarm7_fk_mat_batch_c(double *thetas, double *tcp_mats,
                         int *ret_codes, int batch_size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < batch_size; i++) {
        ret_codes[i] = xarm7_forward_kinematics(
            thetas + i * 7,
            reinterpret_cast<double(*)[4]>(tcp_mats + i * 16));
    }
    return 0;
}

} // extern "C"
