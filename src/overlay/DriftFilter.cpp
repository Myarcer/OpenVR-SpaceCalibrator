#include "DriftFilter.h"

#include <algorithm>
#include <cmath>

using Eigen::Matrix;
using Eigen::Vector3d;
using Vec3  = Eigen::Vector3d;
using Vec6  = Matrix<double, 6, 1>;
using Mat6  = Matrix<double, 6, 6>;
using Mat12 = Matrix<double, 12, 12>;

DriftFilter::DriftFilter() { Reset(); }

void DriftFilter::Reset() {
    T_ = Sophus::SE3d();
    v_.setZero();
    P_.setZero();
    // Initial uncertainty: small on transform (start at identity), larger
    // on velocity (we don't know the drift rate yet).
    for (int i = 0; i < 3; ++i) P_(i, i)         = 0.01 * 0.01;       // 1cm^2
    for (int i = 3; i < 6; ++i) P_(i, i)         = 0.017 * 0.017;     // 1deg^2
    for (int i = 6; i < 9; ++i) P_(i, i)         = 0.05 * 0.05;       // 5cm/s^2
    for (int i = 9; i < 12; ++i) P_(i, i)        = 0.087 * 0.087;     // 5deg/s^2
    high_mahal_streak_ = 0;
    last_mahal_ = 0.0;
    initialized_ = false;
}

void DriftFilter::ResetTo(const Sophus::SE3d& T_init) {
    T_ = T_init;
    v_.setZero();
    P_.setZero();
    // Inflated covariance for fast re-convergence after a reset event.
    for (int i = 0; i < 3; ++i) P_(i, i)         = 0.10 * 0.10;       // 10cm^2
    for (int i = 3; i < 6; ++i) P_(i, i)         = 0.175 * 0.175;     // 10deg^2
    for (int i = 6; i < 9; ++i) P_(i, i)         = 0.20 * 0.20;       // 20cm/s^2
    for (int i = 9; i < 12; ++i) P_(i, i)        = 0.349 * 0.349;     // 20deg/s^2
    high_mahal_streak_ = 0;
    last_mahal_ = 0.0;
    initialized_ = true;
}

void DriftFilter::SymmetrizeP() {
    P_ = 0.5 * (P_ + P_.transpose()).eval();
}

void DriftFilter::Predict(double dt, double user_lin_speed_mps, double user_ang_speed_radps) {
    if (dt <= 0) return;

    // Independent velocity damping for translation and rotation.
    // Translational drift (SLAM scale error during walking) and rotational
    // drift (gyro bias) are independent processes with different dynamics.
    double decay_trans = std::exp(-dt / std::max(1e-3, params.tau_v_trans_s));
    double decay_rot   = std::exp(-dt / std::max(1e-3, params.tau_v_rot_s));

    // Compose per-tick velocity kills into single multipliers, reused below for
    // the F matrix so the covariance shrinks consistently with the applied decay.
    //   ZUPT:           not walking  -> kill v_trans; not rotating -> kill v_rot.
    //   Rotation freeze: rotating    -> kill v_trans HARD (overcorrection fix).
    // During a head turn the lever-arm R-inflation blinds position updates, so a
    // stale v_trans would integrate into overshoot; holding translation steady
    // during rotation mirrors the static FAST preset the user trusts.
    double trans_kill = 1.0;
    if (user_lin_speed_mps < params.zupt_lin_thresh_mps)            trans_kill *= params.zupt_decay_factor;
    if (user_ang_speed_radps > params.rot_vtrans_kill_thresh_radps) trans_kill *= params.rot_vtrans_kill_factor;
    double rot_kill = 1.0;
    if (user_ang_speed_radps < params.zupt_ang_thresh_radps)        rot_kill *= params.zupt_decay_factor;

    double eff_decay_trans = decay_trans * trans_kill;
    double eff_decay_rot   = decay_rot   * rot_kill;
    v_.head<3>() *= eff_decay_trans;
    v_.tail<3>() *= eff_decay_rot;

    // T_pred = T * Exp(v * dt)
    Vec6 increment = v_ * dt;
    Sophus::SE3d delta = Sophus::SE3d::exp(increment);
    T_ = T_ * delta;

    // F = state transition Jacobian (12x12).
    // Block structure:
    //   [ Ad(delta^-1)    J_r * dt ]    where J_r is the right Jacobian of SE(3)
    //   [ 0               I_6      ]
    // For small dt the right-Jacobian J_r ~= I - (1/2) ad(v*dt) + ... but we
    // approximate as I (small-step Euler). Adjoint is exact via Sophus.
    Mat12 F = Mat12::Identity();
    F.block<6, 6>(0, 0) = delta.inverse().Adj();
    F.block<6, 6>(0, 6) = Mat6::Identity() * dt;
    // Velocity block of F reflects the eff_decay_* applied above, so covariance
    // on v shrinks proportionally when v is being damped/killed.
    for (int i = 6; i < 9; ++i)  F(i, i) = eff_decay_trans;
    for (int i = 9; i < 12; ++i) F(i, i) = eff_decay_rot;

    // Q: motion-correlated process noise on the transform block, plus
    //    velocity random walk on the velocity block.
    // Q_transform diagonal scaled by motion magnitude * dt
    Mat12 Q = Mat12::Zero();
    double q_pos = params.sigma_base_pos_sq
                 + params.sigma_lin_pos_sq * user_lin_speed_mps * dt
                 + params.sigma_ang_pos_sq * user_ang_speed_radps * dt;
    double q_rot = params.sigma_base_rot_sq
                 + params.sigma_lin_rot_sq * user_lin_speed_mps * dt
                 + params.sigma_ang_rot_sq * user_ang_speed_radps * dt;
    for (int i = 0; i < 3; ++i) Q(i, i)         = q_pos;
    for (int i = 3; i < 6; ++i) Q(i, i)         = q_rot;
    for (int i = 6; i < 9; ++i) Q(i, i)         = params.sigma_v_pos_sq * dt;
    for (int i = 9; i < 12; ++i) Q(i, i)        = params.sigma_v_rot_sq * dt;

    P_ = F * P_ * F.transpose() + Q;
    SymmetrizeP();
}

bool DriftFilter::Update(const Sophus::SE3d& T_meas,
                         double user_lin_speed_mps,
                         double user_ang_speed_radps,
                         double lever_arm_m,
                         double *out_innov_pos_m,
                         double *out_innov_rot_rad,
                         double *out_mahalanobis,
                         double *out_nis_pos,
                         double *out_nis_rot) {
    if (out_nis_pos) *out_nis_pos = 0.0;
    if (out_nis_rot) *out_nis_rot = 0.0;

    // Bootstrap: snap on first measurement.
    if (!initialized_) {
        T_ = T_meas;
        v_.setZero();
        initialized_ = true;
        if (out_innov_pos_m) *out_innov_pos_m = 0.0;
        if (out_innov_rot_rad) *out_innov_rot_rad = 0.0;
        if (out_mahalanobis) *out_mahalanobis = 0.0;
        last_mahal_ = 0.0;
        return true;
    }

    // Residual on the manifold: y = Log(T_pred^-1 * T_meas)
    Sophus::SE3d residual_T = T_.inverse() * T_meas;
    Vec6 y = residual_T.log();
    Vec3 y_pos = y.head<3>();
    Vec3 y_rot = y.tail<3>();

    if (out_innov_pos_m) *out_innov_pos_m = y_pos.norm();
    if (out_innov_rot_rad) *out_innov_rot_rad = y_rot.norm();  // true measured residual (diagnostic)

    // Rotation channel null (params.correct_rotation == false). Drop the rotation
    // residual from y NOW - before it reaches S, the Mahalanobis test, or the
    // gain - so rotation cannot be corrected, cannot trip a reset on a head-turn
    // transient, and (critically) cannot leak into the position channel through
    // the cross-covariance. The out_innov_rot/out_nis_rot diagnostics above and
    // below still see the true residual via the y_rot copy.
    if (!params.correct_rotation) {
        y.tail<3>().setZero();
    }

    // H: observe T directly (6x12), velocity unobserved per step.
    Matrix<double, 6, 12> H = Matrix<double, 6, 12>::Zero();
    H.block<6, 6>(0, 0) = Mat6::Identity();

    // R: static + omega*L lever-arm inflation + linear-speed*time-skew inflation
    // on position channels. The two motion terms add in quadrature because
    // angular and linear motion are independent error sources.
    Mat6 R = Mat6::Zero();
    double lever_speed = user_ang_speed_radps * lever_arm_m;             // m/s apparent translation from rotation about offset puck
    double skew_disp   = user_lin_speed_mps * params.dt_skew_s;          // m apparent offset from time skew (pos)
    double skew_rot    = user_ang_speed_radps * params.dt_skew_s;        // rad apparent rotation from time skew
    double R_pos_inflated = params.R_static_pos_sq
                          + params.k_lever * lever_speed * lever_speed
                          + params.k_skew  * skew_disp   * skew_disp;
    double R_rot_inflated = params.R_static_rot_sq
                          + params.k_skew_rot  * skew_rot * skew_rot
                          + params.k_omega_rot * user_ang_speed_radps * user_ang_speed_radps;
    for (int i = 0; i < 3; ++i) R(i, i)     = R_pos_inflated;
    for (int i = 3; i < 6; ++i) R(i, i)     = R_rot_inflated;

    Mat6 S = H * P_ * H.transpose() + R;
    Mat6 S_inv = S.inverse();

    // Mahalanobis distance (sqrt of chi-square).
    double mahal_sq = (y.transpose() * S_inv * y)(0, 0);
    double mahal = std::sqrt(std::max(0.0, mahal_sq));
    last_mahal_ = mahal;
    if (out_mahalanobis) *out_mahalanobis = mahal;

    // Per-channel NIS (normalized innovation squared) for the self-tuner.
    // Uses the marginal innovation covariance of each channel (top-left /
    // bottom-right 3x3 of S), which is the correct per-channel consistency
    // statistic. Both are chi-square with mean 3 when the noise model matches.
    if (out_nis_pos) {
        Eigen::Matrix3d S_pos = S.block<3, 3>(0, 0);
        *out_nis_pos = (y_pos.transpose() * S_pos.inverse() * y_pos)(0, 0);
    }
    if (out_nis_rot) {
        Eigen::Matrix3d S_rot = S.block<3, 3>(3, 3);
        *out_nis_rot = (y_rot.transpose() * S_rot.inverse() * y_rot)(0, 0);
    }

    // Reset trigger: sustained large innovation -> snap to measurement.
    if (mahal > params.reset_mahal_thresh) {
        ++high_mahal_streak_;
        if (high_mahal_streak_ >= params.reset_persist_ticks) {
            ResetTo(T_meas);
            reset_event_ = true;
            return true;
        }
        // Don't update during suspected jump - if it's transient, the streak
        // will reset and prediction continues smoothly. If sustained, the
        // ResetTo above kicks in. This avoids contaminating the estimate
        // with what may be a one-off bad sample.
        return true;
    } else {
        high_mahal_streak_ = 0;
    }

    // Standard EKF update.
    Matrix<double, 12, 6> K = P_ * H.transpose() * S_inv;
    Matrix<double, 12, 1> dx = K * y;

    // With y_rot zeroed, K's cross terms would still rotate T_ and drive v_rot
    // from the position residual. Hard-null the rotation tangent of both the
    // transform and velocity updates so the rotation stays locked to its
    // bootstrap value (v_rot then stays 0, so Predict's Exp(v*dt) is pure
    // translation and T_'s rotation never moves).
    if (!params.correct_rotation) {
        dx.segment<3>(3).setZero();   // T rotation tangent
        dx.segment<3>(9).setZero();   // rotation drift velocity
    }

    // Apply on manifold for T (multiplicative), tangent for v.
    Vec6 dT_tangent = dx.head<6>();
    Vec6 dv         = dx.tail<6>();
    T_ = T_ * Sophus::SE3d::exp(dT_tangent);
    v_ = v_ + dv;

    // Joseph form would be more numerically stable; for now standard form
    // followed by symmetrization. Inflated noise floors prevent degeneracy.
    Mat12 IKH = Mat12::Identity() - K * H;
    P_ = IKH * P_;
    SymmetrizeP();

    return true;
}
