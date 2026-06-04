#pragma once

// Principled drift-rate estimator for the SLAM-Fix EKF process noise.
//
// The cross-system calibration offset T_meas drifts as a RANDOM WALK in walked
// distance (translation channel) / accumulated rotation angle (rotation
// channel) - exactly what the EKF's Q assembly assumes:
//     q_pos += sigma_lin_pos_sq * lin_speed * dt    (variance ~ distance)
//     q_rot += sigma_ang_rot_sq * ang_speed * dt    (variance ~ angle)
//
// For a Wiener process observed under white measurement noise R, the structure
// function is LINEAR in the lag L:
//     D(L) = E[ |offset(s+L) - offset(s)|^2 ] = 2*R + sigma_sq * L
// A single least-squares line fit recovers, with NO floors and NO bias:
//     slope     = sigma_sq          (== sigma_lin_pos_sq / sigma_ang_rot_sq)
//     intercept = 2 * R_static      (white measurement-noise variance; a bonus)
//
// This replaces the previous NIS-consistency self-tuner, which entangled Q with
// R: motion inflated R (lever-arm + time-skew terms), which shrank NIS, which
// drove the learned rate DOWN to its clamp floor (~1 cm/m) on exactly the
// walking ticks that carry the drift information. The structure function reads
// the drift directly off the measurement and is immune to that feedback.
//
// Validated on real walk logs: D(L) linear to R^2 = 1.000, slope -> ~5.5 cm/m
// where the old tuner railed to 1 cm/m.

#include <Eigen/Core>
#include <deque>
#include <vector>
#include <cmath>
#include <algorithm>

class DriftStructureEstimator {
public:
	struct Params {
		// Uniform resample grid (offset sampled vs cumulative coordinate).
		double grid_m   = 0.02;      // 2 cm
		double grid_rad = 0.0087;    // ~0.5 deg

		// Line-fit lag window. Skip the shortest lags (jitter / white-noise
		// dominated) and the longest (few sample pairs -> high variance).
		double lag_min_m   = 0.30, lag_max_m   = 2.00, lag_step_m   = 0.05;
		double lag_min_rad = 0.087, lag_max_rad = 0.60, lag_step_rad = 0.0175;

		// Minimum cumulative travel before a fit is trusted.
		double min_span_m   = 5.0;
		double min_span_rad = 1.0;   // ~57 deg

		// Reject fits that are not a clean line (=> not a random walk / bad data).
		double min_r2 = 0.85;

		// SANITY guards only (not tuning floors): refuse absurd slopes. The lower
		// bound sits far below any real working value so it never biases a fit.
		double pos_sq_min = 1e-6, pos_sq_max = 9e-2;   // (0.1 cm/m)^2 .. (30 cm/m)^2
		double rot_sq_min = 1e-7, rot_sq_max = 1e-2;

		// Auto-tune sliding window: keep only the most recent travel so the
		// estimate tracks the current rig state instead of averaging all session.
		double window_span_m   = 40.0;
		double window_span_rad = 30.0;
	};
	Params params;

	void Reset() { pos_ = Chan(); rot_ = Chan(); }

	// Feed one tracking tick. dist_inc / ang_inc is the travel since the previous
	// fed tick (lin_speed*dt / ang_speed*dt). Caller gates on "moving" so the
	// cumulative coordinate is strictly increasing (clean random-walk axis).
	void PushPos(const Eigen::Vector3d& offset_m, double dist_inc_m) {
		if (!(dist_inc_m > 0.0) || !offset_m.allFinite()) return;
		pos_.cum += dist_inc_m;
		pos_.coord.push_back(pos_.cum);
		pos_.val.push_back(offset_m);
	}
	void PushRot(const Eigen::Vector3d& offset_rad, double ang_inc_rad) {
		if (!(ang_inc_rad > 0.0) || !offset_rad.allFinite()) return;
		rot_.cum += ang_inc_rad;
		rot_.coord.push_back(rot_.cum);
		rot_.val.push_back(offset_rad);
	}

	double CumPos() const { return pos_.cum; }
	double CumRot() const { return rot_.cum; }
	double SpanPos() const { return Span(pos_); }
	double SpanRot() const { return Span(rot_); }

	// Drop samples older than window_span behind the latest (auto-tune path).
	void TrimToWindow() {
		Trim(pos_, params.window_span_m);
		Trim(rot_, params.window_span_rad);
	}

	// Fit the translation drift rate. Returns true (and writes sigma_pos_sq, the
	// EKF's sigma_lin_pos_sq in m^2) only when a clean linear structure function
	// over enough travel was found. Optionally returns R_static (m^2), R^2, span.
	bool EstimatePos(double& sigma_pos_sq, double* R_static_pos_sq = nullptr,
	                 double* r2 = nullptr, double* span = nullptr) {
		return Fit(pos_, params.grid_m, params.lag_min_m, params.lag_max_m, params.lag_step_m,
		           params.min_span_m, params.pos_sq_min, params.pos_sq_max,
		           sigma_pos_sq, R_static_pos_sq, r2, span);
	}
	bool EstimateRot(double& sigma_rot_sq, double* r2 = nullptr, double* span = nullptr) {
		return Fit(rot_, params.grid_rad, params.lag_min_rad, params.lag_max_rad, params.lag_step_rad,
		           params.min_span_rad, params.rot_sq_min, params.rot_sq_max,
		           sigma_rot_sq, nullptr, r2, span);
	}

private:
	struct Chan {
		std::deque<double> coord;          // cumulative distance/angle (monotonic up)
		std::deque<Eigen::Vector3d> val;   // measured offset at that coordinate
		double cum = 0.0;
	};
	Chan pos_, rot_;

	static double Span(const Chan& c) {
		if (c.coord.size() < 2) return 0.0;
		return c.coord.back() - c.coord.front();
	}
	static void Trim(Chan& c, double span) {
		if (c.coord.empty()) return;
		double cutoff = c.coord.back() - span;
		while (c.coord.size() > 2 && c.coord.front() < cutoff) {
			c.coord.pop_front();
			c.val.pop_front();
		}
	}

	bool Fit(const Chan& c, double grid, double lagMin, double lagMax, double lagStep,
	         double minSpan, double smin, double smax,
	         double& sigma_sq, double* Rstat, double* r2o, double* spano) const {
		double span = Span(c);
		if (spano) *spano = span;
		if (r2o)   *r2o = 0.0;
		if (span < minSpan || c.coord.size() < 8) return false;

		// Resample each component onto a uniform coordinate grid (linear interp).
		const double x0 = c.coord.front(), x1 = c.coord.back();
		const int N = (int)((x1 - x0) / grid);
		if (N < 4) return false;
		std::vector<Eigen::Vector3d> g(N);
		size_t j = 0;
		for (int i = 0; i < N; ++i) {
			const double x = x0 + i * grid;
			while (j + 1 < c.coord.size() && c.coord[j + 1] < x) ++j;
			const size_t j2 = std::min(j + 1, c.coord.size() - 1);
			const double xa = c.coord[j], xb = c.coord[j2];
			double t = (xb > xa) ? (x - xa) / (xb - xa) : 0.0;
			t = std::clamp(t, 0.0, 1.0);
			g[i] = c.val[j] + t * (c.val[j2] - c.val[j]);
		}

		// Structure function over the lag window.
		std::vector<double> Ls, Ds;
		for (double L = lagMin; L <= lagMax + 1e-9; L += lagStep) {
			const int k = (int)std::lround(L / grid);
			if (k < 1 || k >= N) continue;
			double acc = 0.0; int n = 0;
			for (int i = 0; i + k < N; ++i) { acc += (g[i + k] - g[i]).squaredNorm(); ++n; }
			if (n > 0) { Ls.push_back(L); Ds.push_back(acc / n); }
		}
		if (Ls.size() < 3) return false;

		// Least-squares line  D = a + b*L.
		const double n = (double)Ls.size();
		double sL = 0, sD = 0, sLL = 0, sLD = 0;
		for (size_t i = 0; i < Ls.size(); ++i) {
			sL += Ls[i]; sD += Ds[i]; sLL += Ls[i] * Ls[i]; sLD += Ls[i] * Ds[i];
		}
		const double denom = n * sLL - sL * sL;
		if (std::abs(denom) < 1e-12) return false;
		const double b = (n * sLD - sL * sD) / denom;   // slope  = sigma_sq
		const double a = (sD - b * sL) / n;              // intercept = 2*R

		// Goodness of fit (reject anything that isn't a clean random walk).
		const double meanD = sD / n;
		double ssRes = 0, ssTot = 0;
		for (size_t i = 0; i < Ls.size(); ++i) {
			const double pred = a + b * Ls[i];
			ssRes += (Ds[i] - pred) * (Ds[i] - pred);
			ssTot += (Ds[i] - meanD) * (Ds[i] - meanD);
		}
		const double r2 = (ssTot > 1e-18) ? 1.0 - ssRes / ssTot : 0.0;
		if (r2o) *r2o = r2;
		if (r2 < params.min_r2) return false;
		if (!(b > 0.0)) return false;   // no distance-correlated growth -> keep prior

		sigma_sq = std::clamp(b, smin, smax);
		if (Rstat) *Rstat = std::max(0.0, 0.5 * a);
		return true;
	}
};
