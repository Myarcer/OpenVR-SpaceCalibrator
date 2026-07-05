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
//   P_post = (I - K H) P_pred  (standard form + symmetrization)
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
    // Position-correction persistence ramp g in [0..1] (diagnostic / logging).
    double CorrectionRamp() const { return corr_ramp_; }
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
        // Rotation is POISON for the position channel: T_meas is built from a
        // chain with meter-scale translations (target world pose), so a SLAM
        // rotation lag of omega*dt_skew multiplies into 100s of mm of apparent
        // translation (log 2026-07-05: 813mm innovation at 326 deg/s). These
        // rotation-driven gains must stay high enough to keep the filter blind
        // during head rotation; only the LINEAR skew term is compensated and
        // may run with a low gain.
        double k_lever           = 100.0;    // R-inflation gain on (omega*L)^2
        // Time-skew between reference (SLAM HMD, ~10-25ms latency typical for
        // Pico/Quest streamed via VirtualDesktop/ALVR) and target (lighthouse,
        // low latency) creates apparent translational error proportional to
        // user linear speed. The measurement is now skew-COMPENSATED upstream
        // (CalibrationCalc extrapolates the ref pose forward by dt_skew), so
        // these gains only cover the RESIDUAL skew error; they were 100.0 when
        // R-inflation was the sole defense, which made the filter blind during
        // motion and pushed all correction into the post-move standstill.
        double dt_skew_s         = 0.020;    // assumed time skew (Pico+VD typical)
        double k_skew            = 25.0;     // R-inflation gain on (v_lin*dt_skew)^2
        // Rotation R inflation for residual rotation skew after compensation,
        // plus a per-tick term for raw SLAM rotation jitter that grows during
        // fast rotation (motion blur, feature loss).
        double k_skew_rot        = 100.0;    // R-inflation gain on (omega*dt_skew)^2 (rad^2)
        double k_omega_rot       = 1.0e-3;   // R-inflation gain on omega^2 (rad^2 per (rad/s)^2)

        double reset_mahal_thresh = 5.0;
        int    reset_persist_ticks = 50;     // ~0.5s @ 100Hz

        // --- Persistence ramp (anti small-head-bob catch-up) ---
        // The omega R-inflation only suppresses FAST motion; a SLOW small head
        // bob keeps omega low, so the position gain stays high and the filter
        // chases the transient offset immediately ("SLAM catches up to my head").
        // Fix: only commit to a position correction once the residual PERSISTS in
        // one direction. We low-pass the SIGNED residual: an oscillatory bob
        // cancels out (stays small) while a real net displacement's drift grows
        // it. The ramp gain g in [floor..1] relaxes the position R by 1/g^2, so
        // g~0 -> R huge -> no chase, g=1 -> normal gain. Because g reaches 1.0 for
        // genuine net movement, a real 1-2m walk is corrected at FULL strength -
        // the bob suppression never under-corrects real locomotion.
        double ramp_resid_tau_s     = 0.30;  // EMA time constant on signed residual (dt-normalized)
        double ramp_g_lo_m          = 0.005; // persistent residual <= this -> g at floor (no chase)
        double ramp_g_hi_m          = 0.030; // persistent residual >= this -> g = 1 (full correction)
        double ramp_g_floor         = 0.02;  // min g; 0.10 chased slow sitting bobs (2026-07-05 log),
                                             // real drift leaves the floor via the EMA knee anyway
        double ramp_tau_up_s        = 0.5;   // max ramp-up time toward full gain
        double ramp_tau_down_s      = 0.4;   // faster collapse when the transient ends
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

    // Persistence ramp state.
    Eigen::Vector3d resid_ema_ = Eigen::Vector3d::Zero();  // EMA of signed position residual
    double corr_ramp_ = 0.0;    // g: position-correction gain ramp [0..1]
    double last_dt_   = 0.01;   // dt from the most recent Predict (for ramp slew)

    void SymmetrizeP();
};
