#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Windows.h>
#include <openvr.h>
#include <vector>
#include <deque>

#include "Protocol.h"

enum class CalibrationState
{
	None,
	Begin,
	Rotation,
	Translation,
	Editing,
	Continuous,
	ContinuousStandby,
};

struct StandbyDevice {
	std::string trackingSystem;
	std::string model, serial;
};

struct CalibrationContext
{
	CalibrationState state = CalibrationState::None;
	int32_t referenceID = -1, targetID = -1;

	static const size_t MAX_CONTROLLERS = 8;
	int32_t controllerIDs[MAX_CONTROLLERS];

	StandbyDevice targetStandby, referenceStandby;

	Eigen::Vector3d calibratedRotation;
	Eigen::Vector3d calibratedTranslation;
	double calibratedScale;

	std::string referenceTrackingSystem;
	std::string targetTrackingSystem;

	bool enabled = false;
	bool validProfile = false;
	bool clearOnLog = false;
	bool quashTargetInContinuous = false;
	double timeLastTick = 0, timeLastScan = 0, timeLastAssign = 0;
	bool ignoreOutliers = false;
	double wantedUpdateInterval = 1.0;
	float jitterThreshold = 3.0f;

	bool requireTriggerPressToApply = false;
	bool wasWaitingForTriggers = false;
	bool hasAppliedCalibrationResult = false;

	// SLAM-Fix state (not persisted across sessions except slamFixRLocked)
	bool slamFixRLocked = false; // R saved to profile, bootstrap can be skipped
	int slamFixTrackingTicks = 0; // frames spent in tracking phase this session
	double slamFixLastTickTime = 0.0; // for dt computation in EKF predict
	// Finite-difference velocity state. Many SLAM HMDs (Pico/Quest streaming via
	// VirtualDesktop/ALVR) leave DriverPose_t::vecVelocity at zero - we compute
	// our own from successive HMD pose deltas so motion-Q and lever-arm R
	// inflation actually engage during walking.
	double slamFixHmdPrevX = 0.0, slamFixHmdPrevY = 0.0, slamFixHmdPrevZ = 0.0;
	double slamFixHmdPrevQw = 1.0, slamFixHmdPrevQx = 0.0, slamFixHmdPrevQy = 0.0, slamFixHmdPrevQz = 0.0;
	bool   slamFixHmdPrevValid = false;
	// EMA-smoothed velocity magnitudes. Raw finite-diff is spiky (SLAM jitter,
	// frame-time variance, pose buffer staleness) - one 245 m/s spike pumps
	// process noise huge for many ticks, causing snapping. EMA + hard clamp
	// turns single-frame outliers into a small bump.
	double slamFixVelLinEma = 0.0;
	double slamFixVelAngEma = 0.0;
	// Periodic R_mount refinement counter. Every N tracking ticks, re-run
	// pose averaging to keep R_mount fresh (prevents frozen mount error from
	// amplifying into apparent translation during head rotation).
	int slamFixRMountRefineTicks = 0;
	// Kabsch recenter: periodic evaluation budget (~1s cadence). Gated purely
	// by confidence (variance + RMS error + improvement over current EKF state),
	// not motion speed. Translation-only fallback when rotation is not observable.
	int slamFixKabschRecenterTicks = 0;
	// Settle gate for the Kabsch recenter: timestamp of the last tick with the
	// HMD in motion. While the user moves, the sample buffer carries
	// lever-arm/skew artifacts (NOT drift) - a recenter fired mid-motion snaps
	// to a corrupted fit (2026-07-05 log: errCal 20mm -> 166mm after a recenter
	// at 63 deg/s head speed). Only recenter after ~1.5s of near-stillness.
	double slamFixLastMotionTime = 0.0;


	float xprev, yprev, zprev;

	float continuousCalibrationThreshold;
	float maxRelativeErrorThreshold = 0.005f;
	Eigen::Vector3d continuousCalibrationOffset;

	protocol::AlignmentSpeedParams alignmentSpeedParams;
	bool enableStaticRecalibration;
	bool lockRelativePosition = false;

	Eigen::AffineCompact3d refToTargetPose = Eigen::AffineCompact3d::Identity();
	bool relativePosCalibrated = false;

	enum Speed
	{
		SLAM_FIX = -1,
		FAST = 0,
		SLOW = 1,
		VERY_SLOW = 2
	};
	Speed calibrationSpeed = FAST;

	vr::DriverPose_t devicePoses[vr::k_unMaxTrackedDeviceCount];

	CalibrationContext() {
		calibratedScale = 1.0;
		memset(devicePoses, 0, sizeof(devicePoses));
		ResetConfig();
	}

	void ResetConfig() {
		alignmentSpeedParams.thr_rot_tiny = 0.49f * (EIGEN_PI / 180.0f);
		alignmentSpeedParams.thr_rot_small = 0.5f * (EIGEN_PI / 180.0f);
		alignmentSpeedParams.thr_rot_large = 5.0f * (EIGEN_PI / 180.0f);

		alignmentSpeedParams.thr_trans_tiny = 0.98f / 1000.0; // mm
		alignmentSpeedParams.thr_trans_small = 1.0f / 1000.0; // mm
		alignmentSpeedParams.thr_trans_large = 20.0f / 1000.0; // mm

		alignmentSpeedParams.align_speed_tiny = 1.0f;
		alignmentSpeedParams.align_speed_small = 1.0f;
		alignmentSpeedParams.align_speed_large = 2.0f;

		continuousCalibrationThreshold = 1.5f;
		maxRelativeErrorThreshold = 0.005f;
		jitterThreshold = 3.0f;

		continuousCalibrationOffset = Eigen::Vector3d::Zero();

		enableStaticRecalibration = false;
	}

	struct Chaperone
	{
		bool valid = false;
		bool autoApply = true;
		std::vector<vr::HmdQuad_t> geometry;
		vr::HmdMatrix34_t standingCenter = {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
		};
		vr::HmdVector2_t playSpaceSize = { 0.0f, 0.0f };
	} chaperone;

	void ClearLogOnMessage() {
		clearOnLog = true;
	}

	void Clear()
	{
		chaperone.geometry.clear();
		chaperone.standingCenter = vr::HmdMatrix34_t();
		chaperone.playSpaceSize = vr::HmdVector2_t();
		chaperone.valid = false;

		calibratedRotation = Eigen::Vector3d();
		calibratedTranslation = Eigen::Vector3d();
		calibratedScale = 1.0;
		referenceTrackingSystem = "";
		targetTrackingSystem = "";
		enabled = false;
		validProfile = false;
		refToTargetPose = Eigen::AffineCompact3d::Identity();
		relativePosCalibrated = true;
	}

	size_t SampleCount()
	{
		switch (calibrationSpeed)
		{
		case SLAM_FIX:
			// Bootstrap window for SLAM-Fix mode. Only used until the rigid
			// device-to-device offset (m_refToTargetPose) is locked - after
			// that, SLAM-Fix runs a per-frame manifold low-pass filter and
			// does not depend on this window size.
			return 120;
		case FAST:
			return 100;
		case SLOW:
			return 250;
		case VERY_SLOW:
			return 500;
		}
		return 100;
	}

	bool IsSlamFix() const { return calibrationSpeed == SLAM_FIX; }

	// SLAM-Fix runtime overrides. These return SLAM-Fix-specific values when
	// SLAM-Fix is the active speed, otherwise the user-configured value. They
	// never mutate the persisted profile fields - the user's saved settings are
	// preserved and restored when switching back to FAST/SLOW/VERY_SLOW.
	//
	// SLAM-Fix has two phases:
	//   1) BOOTSTRAP - the existing Kabsch sliding-window path runs, with
	//      these tuning values, until the rigid relative pose (R) locks.
	//   2) TRACKING - per-frame closed-form T_meas = ref * R * target^-1,
	//      blended via low-pass filter on SE(3). Bypasses the Kabsch path.
	bool EffectiveLockRelativePosition() const {
		// Never force lock during bootstrap - we need R to converge naturally.
		return IsSlamFix() ? false : lockRelativePosition;
	}
	bool EffectiveStaticRecalibration() const {
		return IsSlamFix() ? true : enableStaticRecalibration;
	}
	bool EffectiveIgnoreOutliers() const {
		return IsSlamFix() ? true : ignoreOutliers;
	}
	float EffectiveContinuousCalibrationThreshold() const {
		return IsSlamFix() ? 2.0f : continuousCalibrationThreshold;
	}
	float EffectiveMaxRelativeErrorThreshold() const {
		return IsSlamFix() ? 0.025f : maxRelativeErrorThreshold;
	}
	float EffectiveJitterThreshold() const {
		return IsSlamFix() ? 5.0f : jitterThreshold;
	}
	protocol::AlignmentSpeedParams EffectiveAlignmentSpeedParams() const {
		if (!IsSlamFix()) return alignmentSpeedParams;
		protocol::AlignmentSpeedParams p;
		// Aggressive blend: visual snap once a new cal is accepted.
		// In TRACKING phase the LPF itself controls smoothing; these only
		// matter for the rare large innovations that get applied.
		// Faster drift recovery: the EKF posterior is already smoothed, so the
		// driver's extra lerp was redundant latency (~67ms tau at large=15).
		// Bumped to cut the visible drift-then-correct lag roughly in half.
		p.align_speed_tiny = 4.0f;
		p.align_speed_small = 15.0f;
		p.align_speed_large = 30.0f;
		p.thr_trans_tiny = 0.5f / 1000.0f;
		p.thr_trans_small = 1.0f / 1000.0f;
		p.thr_trans_large = 5.0f / 1000.0f;
		p.thr_rot_tiny = 0.2f * (float)(EIGEN_PI / 180.0);
		p.thr_rot_small = 0.5f * (float)(EIGEN_PI / 180.0);
		p.thr_rot_large = 2.0f * (float)(EIGEN_PI / 180.0);
		return p;
	}

	struct Message
	{
		enum Type
		{
			String,
			Progress
		} type = String;

		Message(Type type) : type(type), progress(0), target(0) { }

		std::string str;
		int progress, target;
	};

	std::deque<Message> messages;

	void Log(const std::string &msg)
	{
		if (clearOnLog) {
			messages.clear();
			clearOnLog = false;
		}

		if (messages.empty() || messages.back().type == Message::Progress)
			messages.push_back(Message(Message::String));

		OutputDebugStringA(msg.c_str());

		messages.back().str += msg;
		std::cerr << msg;

		while (messages.size() > 15) messages.pop_front();
	}

	void Progress(int current, int target)
	{
		if (messages.empty() || messages.back().type == Message::String)
			messages.push_back(Message(Message::Progress));

		messages.back().progress = current;
		messages.back().target = target;
	}

	bool TargetPoseIsValidSimple() const {
		return targetID >= 0 && targetID <= vr::k_unMaxTrackedDeviceCount
			&& devicePoses[targetID].poseIsValid && devicePoses[targetID].result == vr::ETrackingResult::TrackingResult_Running_OK;
	}

	bool ReferencePoseIsValidSimple() const {
		return referenceID >= 0 && referenceID <= vr::k_unMaxTrackedDeviceCount
			&& devicePoses[referenceID].poseIsValid && devicePoses[referenceID].result == vr::ETrackingResult::TrackingResult_Running_OK;
	}
};

extern CalibrationContext CalCtx;

void InitCalibrator();
void CalibrationTick(double time);
void StartCalibration();
void StartContinuousCalibration();
void EndContinuousCalibration();
void LoadChaperoneBounds();
void ApplyChaperoneBounds();

void PushCalibrationApplyTime();
void ShowCalibrationDebug(int r, int c);
void DebugApplyRandomOffset();