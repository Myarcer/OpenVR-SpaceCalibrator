// Self-tuning estimator for the SLAM-Fix EKF process-noise (drift rate).
//
// Physical mapping (see DriftFilter.cpp Q assembly):
//   process noise accumulated over walked distance L is sigma_lin_pos_sq * L,
//   so the filter's expected drift std over 1m is sqrt(sigma_lin_pos_sq).
//   => sigma_lin_pos_sq == (drift_per_meter)^2
//      sigma_ang_rot_sq == (drift_per_rad)^2
//
// We never measure drift-per-meter directly; instead we use innovation
// consistency. Per EKF update the normalized innovation squared
//   NIS = y^T S^-1 y
// is chi-square with mean = DOF (3 per channel) when the noise model matches
// reality. If the robust median of NIS_channel exceeds DOF the filter is
// over-confident (Q too small for this headset) -> scale sigma up; if below,
// scale down. Median + MAD gating gives the outlier rejection SLAM needs
// (teleports / reset ticks produce huge one-off NIS the median ignores).
//
// Two channels (translation, rotation) are tuned independently because the EKF
// treats them as independent processes (SLAM scale error vs gyro bias).

#pragma once

#include <array>
#include <cstddef>

class DriftRateTuner {
public:
	struct Params {
		// Auto path: exponent on the NIS ratio per window. Small => slow.
		double learn_gain = 0.25;
		// Minimum collected samples before an auto application fires (~a window).
		int    min_samples = 300;
		// Outlier gate: drop samples with |nis - median| > mad_factor * MAD.
		double mad_factor = 5.0;
		// Per-window multiplicative step clamp (auto path).
		double step_clamp_lo = 0.95;
		double step_clamp_hi = 1.05;
		// One-shot step clamp (manual snap path) - allows a big correction but
		// guards against absurd values from a degenerate walk.
		double snap_clamp_lo = 0.1;
		double snap_clamp_hi = 10.0;
		// Sigma clamps (variance). pos in m^2 = (m/m)^2, rot in rad^2 = (rad/rad)^2.
		double pos_sq_min = 1e-4;    // (1cm/m)^2
		double pos_sq_max = 6.25e-2; // (25cm/m)^2
		double rot_sq_min = 1e-6;
		double rot_sq_max = 4e-3;
	};
	Params params;

	DriftRateTuner();
	void Reset();

	// Gated per-tick NIS sample for each channel. `moving` must be true (no
	// motion => no drift information). Caller is responsible for skipping
	// bootstrap / reset ticks.
	void PushPos(double nis, bool moving);
	void PushRot(double nis, bool moving);

	// Auto path: once a window's worth of samples is collected, nudge sigma by
	// a small multiplicative step toward NIS-consistency and clear the buffer.
	// Returns true (and writes sigma_sq) if applied.
	bool MaybeApplyPos(double& sigma_pos_sq);
	bool MaybeApplyRot(double& sigma_rot_sq);

	// Manual path: snap sigma to the full median-implied correction now,
	// regardless of sample count (uses whatever is buffered). Clears the buffer.
	bool SnapPos(double& sigma_pos_sq);
	bool SnapRot(double& sigma_rot_sq);

	std::size_t CountPos() const { return pos_.count; }
	std::size_t CountRot() const { return rot_.count; }

private:
	static const int DOF = 3;
	static const std::size_t N = 600;

	struct Channel {
		std::array<float, N> buf{};
		std::size_t idx = 0;
		std::size_t count = 0;
		void push(double v) {
			buf[idx] = static_cast<float>(v);
			idx = (idx + 1) % N;
			if (count < N) ++count;
		}
		void clear() { idx = 0; count = 0; }
	};
	Channel pos_, rot_;

	// Robust median of buffered samples after MAD outlier rejection.
	// Returns < 0 if too few samples.
	double RobustMedian(const Channel& c) const;

	// Shared apply logic. `snap` selects full vs slow step + the relevant clamp.
	// Returns true if sigma was updated.
	bool Apply(Channel& c, double& sigma_sq, double smin, double smax, bool snap);
};
