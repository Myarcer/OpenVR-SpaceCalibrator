// SLAM-Fix v2: SE(3) Extended Kalman Filter for SLAM drift correction.
//
// State (12-dim):
//   T  in SE(3)  - current ref->target drift transform (manifold)
//   v  in se(3)  - drift velocity, 6-vector (3 lin, 3 ang) in tangent space
//
// Predict (per tick, dt ~10ms @ 100Hz):
//   T_pred = T * Exp(v * dt)
//   v_pred = v
//   P_pred = F * P * F^T + Q(motion)
//
// Update (per measurement):
//   T_meas = ref * R_mount * target^-1
//   y      = Log(T_pred^-1 * T_meas)        (6-vector residual)
//   S      = H * P_pred * H^T + R(omega, L)
//   K      = P_pred * H^T * S^-1
//   dx     = K * y
//   T_post = T_pred * Exp(dx[0:6])
//   v_post = v_pred + dx[6:12]
//   P_post = (I - K H) P_pred  (Joseph form for stability)
//
// Reset (Mahalanobis-driven):
//   m = sqrt(y^T * S^-1 * y)
//   if m > thresh sustained for > N ticks:
//     T = T_meas, v = 0, P = inflated
//
// See plan.md for tuning rationale.

#pragma once

#include <Eigen/Dense>
#include <sophus/se3.hpp>

class DriftFilter {
public:
    DriftFilter();

    // Reset to identity drift, zero velocity, default covariance.
    void Reset();

    // Reset to a known transform (e.g. snap to current measurement after a
    // detected re-seat / large-innovation event). Inflates covariance so the
    // filter re-converges quickly.
    void ResetTo(const Sophus::SE3d& T_init);

    // Predict step. user_lin_speed_mps is ||linear velocity of HMD in world||,
    // user_ang_speed_radps is ||angular velocity of HMD||.
    // Both are used to scale the process noise so the filter trusts SLAM less
    // when the user is moving (which is when drift accumulates).
    void Predict(double dt, double user_lin_speed_mps, double user_ang_speed_radps);

    // Update step.
    //   T_meas: the per-frame measurement T_ref * R_mount * T_target^-1
    //   user_ang_speed_radps: HMD angular speed (for R inflation, lever arm)
    //   lever_arm_m: scalar magnitude of puck-to-HMD offset (default 0.10m)
    // Returns true if update applied. Always populates innovations.
    bool Update(const Sophus::SE3d& T_meas,
                double user_lin_speed_mps,
                double user_ang_speed_radps,
                double lever_arm_m,
                double *out_innov_pos_m,
                double *out_innov_rot_rad,
                double *out_mahalanobis);

    // Current drift transform estimate.
    const Sophus::SE3d& Transform() const { return T_; }

    // Current drift velocity in tangent space (3 lin, 3 ang).
    Eigen::Matrix<double, 6, 1> Velocity() const { return v_; }

    // For logging.
    double LastMahalanobis() const { return last_mahal_; }
    int    HighMahalStreak() const { return high_mahal_streak_; }
    bool   ConsumeResetEvent() { bool e = reset_event_; reset_event_ = false; return e; }
    bool IsInitialized() const { return initialized_; }
    void MarkInitialized() { initialized_ = true; }

    // Tuning (defaults set in ctor, can be overridden externally if needed).
    // All variances; sigmas squared.
    struct Params {
        double sigma_base_pos_sq = 1e-8;     // m^2 baseline (very small)
        double sigma_base_rot_sq = 1e-8;     // rad^2 baseline
        double sigma_lin_pos_sq  = 2.5e-3;   // (~5cm/sqrt(m))^2 walk drift
        double sigma_lin_rot_sq  = 1e-4;     // tiny rotational drift coupling
        double sigma_ang_pos_sq  = 1e-5;     // tiny pos drift from rotation
        double sigma_ang_rot_sq  = 1e-4;     // (~0.01 rad/sqrt(rad))^2
        // Velocity random walk noise (per second).
        double sigma_v_pos_sq    = 1e-4;
        double sigma_v_rot_sq    = 1e-4;
        // Velocity damping time constants. Translational and rotational drift
        // velocity decay independently - they arise from different physical
        // processes (SLAM scale error vs gyro bias) and must not cross-couple.
        double tau_v_trans_s     = 1.5;  // translational v decay (s)
        double tau_v_rot_s       = 3.0;  // rotational v decay (s)

        // ZUPT (Zero Velocity Update) threshold. When user linear speed
        // drops below this, translational v is aggressively decayed to
        // prevent walking-acquired v_trans from integrating during
        // subsequent rotation (where R-inflation suppresses real updates).
        double zupt_lin_thresh_mps = 0.05;  // m/s - below this = "not walking"
        double zupt_ang_thresh_radps = 0.1; // rad/s - below this = "not rotating"
        double zupt_decay_factor = 0.85;    // per-tick multiplier (~15% kill/tick @ 100Hz, ~100ms to near-zero)

        // Rotation freeze (rotation-overcorrection fix). During a head turn the
        // lever-arm R-inflation blinds the position channel, so any residual
        // translational drift velocity integrates unchecked into overshoot - the
        // SLAM preset's visible "overcorrects on rotation" vs the static FAST
        // preset. Above this angular speed, kill v_trans HARD so the transform
        // HOLDS during rotation instead of coasting on a stale velocity.
        double rot_vtrans_kill_thresh_radps = 0.35; // rad/s (~20deg/s) = clearly rotating
        double rot_vtrans_kill_factor       = 0.5;  // per-tick v_trans multiplier while rotating

        double R_static_pos_sq   = 2.5e-5;   // (5mm)^2 lighthouse position noise
        double R_static_rot_sq   = 2.7e-3;   // (3deg)^2 - per-sample SLAM rotation noise
                                             // floor. Per-frame EKF needs much higher
                                             // static floor than FAST preset's Kabsch
                                             // (which averages 100-500 samples,
                                             // effectively shrinking sigma by sqrt(N)).
        double k_lever           = 100.0;    // R-inflation gain on (omega*L)^2
        // Time-skew between reference (lighthouse, low latency) and target
        // (SLAM, ~10-25ms latency typical for Pico/Quest streamed via
        // VirtualDesktop/ALVR) creates apparent translational error
        // proportional to user linear speed. R_pos += k_skew*(v_lin*dt_skew)^2
        // absorbs this without rejecting samples.
        double dt_skew_s         = 0.020;    // assumed time skew (Pico+VD typical)
        double k_skew            = 100.0;    // R-inflation gain on (v_lin*dt_skew)^2
        // Rotation R inflation. Lighthouse rotation is fast; SLAM rotation
        // lags by dt_skew so during a head turn there is a systematic
        // rotation residual ~ omega*dt_skew that must NOT trigger aggressive
        // correction. Plus a per-tick term for raw SLAM rotation jitter
        // that grows during fast rotation (motion blur, feature loss).
        double k_skew_rot        = 100.0;    // R-inflation gain on (omega*dt_skew)^2 (rad^2)
        double k_omega_rot       = 1.0e-3;   // R-inflation gain on omega^2 (rad^2 per (rad/s)^2)

        double reset_mahal_thresh = 5.0;
        int    reset_persist_ticks = 50;     // ~0.5s @ 100Hz
    };
    Params params;

private:
    Sophus::SE3d T_;                        // drift transform
    Eigen::Matrix<double, 6, 1> v_;         // drift velocity (tangent)
    Eigen::Matrix<double, 12, 12> P_;       // covariance

    int high_mahal_streak_ = 0;
    double last_mahal_ = 0.0;
    bool initialized_ = false;
    bool reset_event_ = false;  // set true on the tick that ResetTo fires

    void SymmetrizeP();
};
