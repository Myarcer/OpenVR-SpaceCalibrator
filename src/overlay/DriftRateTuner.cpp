#include "DriftRateTuner.h"

#include <algorithm>
#include <cmath>
#include <vector>

DriftRateTuner::DriftRateTuner() { Reset(); }

void DriftRateTuner::Reset() {
	pos_.clear();
	rot_.clear();
}

void DriftRateTuner::PushPos(double nis, bool moving) {
	if (!moving) return;
	if (!std::isfinite(nis) || nis < 0.0) return;
	pos_.push(nis);
}

void DriftRateTuner::PushRot(double nis, bool moving) {
	if (!moving) return;
	if (!std::isfinite(nis) || nis < 0.0) return;
	rot_.push(nis);
}

double DriftRateTuner::RobustMedian(const Channel& c) const {
	if (c.count < 8) return -1.0;

	std::vector<float> v(c.buf.begin(), c.buf.begin() + c.count);

	auto median_of = [](std::vector<float>& s) -> double {
		if (s.empty()) return -1.0;
		size_t mid = s.size() / 2;
		std::nth_element(s.begin(), s.begin() + mid, s.end());
		double m = s[mid];
		if (s.size() % 2 == 0) {
			// average of the two central elements for an even count
			std::nth_element(s.begin(), s.begin() + mid - 1, s.end());
			m = 0.5 * (m + s[mid - 1]);
		}
		return m;
	};

	double med = median_of(v);
	if (med < 0.0) return -1.0;

	// MAD-based outlier rejection: keep |x - med| <= mad_factor * MAD.
	std::vector<float> dev(v.size());
	for (size_t i = 0; i < v.size(); ++i) dev[i] = std::fabs(v[i] - (float)med);
	double mad = median_of(dev);

	if (mad > 1e-9) {
		std::vector<float> inliers;
		inliers.reserve(v.size());
		double cap = params.mad_factor * mad;
		for (float x : v) if (std::fabs(x - med) <= cap) inliers.push_back(x);
		if (inliers.size() >= 4) med = median_of(inliers);
	}

	return med;
}

bool DriftRateTuner::Apply(Channel& c, double& sigma_sq, double smin, double smax, bool snap) {
	double med = RobustMedian(c);
	if (med < 0.0) return false;

	double ratio = med / static_cast<double>(DOF);

	double step;
	if (snap) {
		step = std::clamp(ratio, params.snap_clamp_lo, params.snap_clamp_hi);
	} else {
		step = std::pow(ratio, params.learn_gain);
		step = std::clamp(step, params.step_clamp_lo, params.step_clamp_hi);
	}

	sigma_sq = std::clamp(sigma_sq * step, smin, smax);
	c.clear();
	return true;
}

bool DriftRateTuner::MaybeApplyPos(double& sigma_pos_sq) {
	if (pos_.count < (std::size_t)params.min_samples) return false;
	return Apply(pos_, sigma_pos_sq, params.pos_sq_min, params.pos_sq_max, false);
}

bool DriftRateTuner::MaybeApplyRot(double& sigma_rot_sq) {
	if (rot_.count < (std::size_t)params.min_samples) return false;
	return Apply(rot_, sigma_rot_sq, params.rot_sq_min, params.rot_sq_max, false);
}

bool DriftRateTuner::SnapPos(double& sigma_pos_sq) {
	return Apply(pos_, sigma_pos_sq, params.pos_sq_min, params.pos_sq_max, true);
}

bool DriftRateTuner::SnapRot(double& sigma_rot_sq) {
	return Apply(rot_, sigma_rot_sq, params.rot_sq_min, params.rot_sq_max, true);
}
