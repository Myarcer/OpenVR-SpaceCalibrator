#pragma once

#include <Eigen/Dense>
#include <openvr.h>
#include <vector>
#include <deque>
#include <iostream>
#include <memory>

class DriftFilter;

struct Pose
{
	Eigen::Matrix3d rot;
	Eigen::Vector3d trans;

	Pose() { }
	Pose(const Eigen::AffineCompact3d& transform) {
		rot = transform.rotation();
		trans = transform.translation();
	}
	
	Pose(vr::HmdMatrix34_t hmdMatrix)
	{
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				rot(i, j) = hmdMatrix.m[i][j];
			}
		}
		trans = Eigen::Vector3d(hmdMatrix.m[0][3], hmdMatrix.m[1][3], hmdMatrix.m[2][3]);
	}
	Pose(vr::HmdQuaternion_t rot, const double *trans) {
		this->rot = Eigen::Matrix3d(Eigen::Quaterniond(rot.w, rot.x, rot.y, rot.z));
		this->trans = Eigen::Vector3d(trans[0], trans[1], trans[2]);
	}
	Pose(double x, double y, double z) : trans(Eigen::Vector3d(x, y, z)) { }

	Eigen::Matrix4d ToAffine() const {
		Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				matrix(i, j) = rot(i, j);
			}
			matrix(i, 3) = trans(i);
		}

		return matrix;
	}
};

struct Sample
{
	Pose ref, target;
	bool valid;
	double timestamp;
	Sample() : valid(false), timestamp(0) { }
	Sample(Pose ref, Pose target, double timestamp) : valid(true), ref(ref), target(target), timestamp(timestamp){ }
};

class CalibrationCalc {
public:
	static const double AxisVarianceThreshold;

	bool enableStaticRecalibration;
	bool lockRelativePosition = false;
	
	const Eigen::AffineCompact3d Transformation() const 
	{
		return m_estimatedTransformation;
	}

	const Eigen::Vector3d EulerRotation() const {
		auto rot = m_estimatedTransformation.rotation();
		return rot.eulerAngles(2, 1, 0) * 180.0 / EIGEN_PI;
	}

	bool isValid() const {
		return m_isValid;
	}
	
	const Eigen::AffineCompact3d RelativeTransformation() const 
	{
		return m_refToTargetPose;
	}

	bool isRelativeTransformationCalibrated() const
	{
		return m_relativePosCalibrated;
	}

	void setRelativeTransformation(const Eigen::AffineCompact3d transform, bool calibrated)
	{
		m_refToTargetPose = transform;
		m_relativePosCalibrated = calibrated;
	}

	void PushSample(const Sample& sample);
	void Clear();

	double ReferenceJitter() const;
	double TargetJitter() const;

	bool ComputeOneshot(const bool ignoreOutliers);
	bool ComputeIncremental(bool &lerp, double threshold, double relPoseMaxError, const bool ignoreOutliers);

	// SLAM-Fix per-frame SE(3) EKF drift tracker.
	// Requires m_relativePosCalibrated == true (rigid offset already locked).
	// Computes T_meas = ref * R * target^-1 from the most recent sample, runs
	// EKF predict (with motion-correlated Q) + update (with omega*L-inflated R)
	// and writes the posterior to m_estimatedTransformation.
	//   dt: seconds since last call (clamped to [0.001, 0.1])
	//   user_lin_speed_mps / user_ang_speed_radps: HMD velocity magnitudes
	//   lever_arm_m: scalar puck-to-HMD offset (default 0.10m)
	// Output params filled regardless of return: innovation magnitudes and
	// Mahalanobis distance for logging / phase classification.
	// Returns false if a sample is unavailable or R is not yet locked.
	// Q (process noise) uses smoothed velocities so a single SLAM glitch
	// doesn't pump P; R (measurement noise) uses raw clamped velocities so
	// inflation reacts immediately at the start of a head turn instead of
	// lagging by the EMA time constant.
	bool SlamFixDriftStep(double dt,
		double lin_speed_q_mps, double ang_speed_q_radps,
		double lin_speed_r_mps, double ang_speed_r_radps,
		double lever_arm_m,
		double *innovation_pos_m, double *innovation_rot_rad,
		double *mahalanobis);

	// Force-reset the EKF (e.g. on user request or after calibration mode change).
	void SlamFixDriftReset();

	// Returns true exactly once per actual reset event (sustained Mahalanobis
	// trip that snapped state to T_meas). Used for log phase classification.
	bool SlamFixConsumeResetEvent();

	// Internal EKF state for diagnostics (Mahalanobis, streak).
	double SlamFixLastMahalanobis() const;
	// Position-correction persistence ramp g in [0..1] (logging/diagnostic).
	double SlamFixCorrectionRamp() const;

	// Periodic R_mount refinement. Re-runs pose averaging on a sliding window
	// of recent samples to update m_refToTargetPose, preventing frozen mount
	// error from amplifying into apparent translation during head rotation.
	// Returns true if R_mount was updated.
	// blend_alpha: low-pass factor for small corrections (0=ignore, 1=snap)
	// max_pos_delta_m: position threshold - larger delta triggers snap+reset
	// max_rot_delta_rad: rotation threshold - larger delta triggers snap+reset
	bool RefineRMount(double blend_alpha = 0.15,
	                  double max_pos_delta_m = 0.02,
	                  double max_rot_delta_rad = 0.035);

	// Kabsch recenter for SLAM-Fix. Mirrors FAST's full-Kabsch acceptance logic
	// in ComputeIncremental exactly: variance gate (skip if low AND declining),
	// ValidateCalibration (100mm), improvement gate (threshold, caller passes
	// FAST's contThr=1.4 requiring 28% improvement). Only fires when the buffer
	// genuinely supports a confident re-solve — same conditions FAST uses.
	// out_axisVariance: written with computed axis variance for metrics.
	// Returns true if a correction was applied.
	bool SlamFixKabschRecenter(bool ignoreOutliers, double threshold, double* out_axisVariance = nullptr);

	// Compute current calibration quality metrics without modifying state.
	void ComputeCurrentCalMetrics(double* rmsError, Eigen::Vector3d* posOffset) const;

	size_t SampleCount() const {
		return m_samples.size();
	}

	void ShiftSample() {
		if (!m_samples.empty()) m_samples.pop_front();
	}

	CalibrationCalc();
	~CalibrationCalc();

	// Debug fields
	Eigen::Vector3d m_posOffset;
	double m_axisVariance = 0.0;
	long m_calcCycle;

private:
	bool m_isValid;
	Eigen::AffineCompact3d m_estimatedTransformation;
	bool m_relativePosCalibrated = false;

	/*
	 * This affine transform estimates the pose of the target within the reference device's local pose space.
	 * That is to say, it's given by transforming the target world pose by the inverse reference pose.
	 */
	Eigen::AffineCompact3d m_refToTargetPose = Eigen::AffineCompact3d::Identity();

	std::unique_ptr<DriftFilter> m_driftFilter;

	// SLAM-Fix time-skew compensation: finite-difference LINEAR velocity of the
	// ref (SLAM) device, used to extrapolate its position forward by dt_skew so
	// the ref/target pair is time-aligned before forming T_meas. Rotation is
	// intentionally not extrapolated (see SlamFixDriftStep).
	bool m_slamFixPrevRefValid = false;
	Eigen::Vector3d m_slamFixPrevRefPos = Eigen::Vector3d::Zero();
	Eigen::Vector3d m_slamFixRefVelLin = Eigen::Vector3d::Zero();

	std::deque<Sample> m_samples;

	std::vector<bool> DetectOutliers() const;
	Eigen::Vector3d CalibrateRotation(const bool ignoreOutliers) const;
	Eigen::Vector3d CalibrateTranslation(const Eigen::Matrix3d &rotation) const;
	void CalibrateScaleOffset(const Eigen::Matrix3d &rotation, Eigen::Vector3d* out_scaleOffset, float* out_scaleFactor) const;

	Eigen::AffineCompact3d ComputeCalibration(const bool ignoreOutliers) const;

	double RetargetingErrorRMS(const Eigen::Vector3d& hmdToTargetPos, const Eigen::AffineCompact3d& calibration) const;
	Eigen::Vector3d ComputeRefToTargetOffset(const Eigen::AffineCompact3d& calibration) const;

	Eigen::Vector4d ComputeAxisVariance(const Eigen::AffineCompact3d& calibration) const;

	[[nodiscard]] bool ValidateCalibration(const Eigen::AffineCompact3d& calibration, double *errorOut = nullptr, Eigen::Vector3d* posOffsetV = nullptr);
	void ComputeInstantOffset();

	Eigen::AffineCompact3d EstimateRefToTargetPose(const Eigen::AffineCompact3d& calibration) const;
	bool CalibrateByRelPose(Eigen::AffineCompact3d &out) const;
};