#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Windows.h>
#include <openvr.h>
#include <vector>
#include <deque>

#include "Protocol.h"
#include "DriftRateTuner.h"

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
	// Kabsch recenter: periodic timer replaces motion/still state machine.
	// Runs every N ticks regardless of velocity — axis variance + RMS
	// validation gate quality. Translation-only fallback when variance is low.
	int slamFixKabschRecenterTicks = 0;

	// --- SLAM-Fix self-tuning drift rate ---
	// Learned EKF process-noise (variances). sigma_lin_pos_sq == (drift_per_meter)^2,
	// sigma_ang_rot_sq == (drift_per_rad)^2. Persisted; pushed into the filter at
	// calibration start via CalibrationCalc::SlamFixSetDriftRates. Defaults match
	// the filter's built-in (5cm/m, ~0.01 rad/rad).
	double slamFixDriftPosSq = 2.5e-3;
	double slamFixDriftRotSq = 1e-4;
	double slamFixTimeSkew   = 0.0;     // assumed streaming latency (s); 0 = snappiest (default)
	// Per-axis SLAM scale gain (x,y,z), measured per-headset over a calibration walk. {1,1,1}
	// = identity. >1 means PICO over-reports motion along that axis; applied driver-side as a
	// rigid per-distance rig shift that cancels the anisotropic scale error as you walk.
	Eigen::Vector3d slamFixScale = Eigen::Vector3d::Ones();
	bool   slamFixAutoTune   = false;   // continuous refinement (off until seeded)
	bool   slamFixDriftSeeded = false;  // a manual walk (or load) has set a real rate
	// Auto-tuner controls (exposed in Settings, persisted).
	float  slamFixTuneLearnGain = 0.25f;
	int    slamFixTuneMinSamples = 300;
	float  slamFixTuneMadFactor = 5.0f;
	float  slamFixWalkDurationS = 15.0f;
	// Manual-walk runtime state. The walk runs inside the normal Continuous
	// state (no separate CalibrationState) - this flag selects aggressive
	// collection + snap-on-completion instead of slow auto refinement.
	bool   slamFixWalkActive = false;
	double slamFixWalkStartTime = 0.0;
	DriftRateTuner slamFixTuner;

	// --- One-time latency (time-skew) calibration ---
	// Measures the reference<->SLAM streaming latency by cross-correlating the two
	// devices' angular-speed signals during a brisk yaw head-shake, replacing the
	// guessed/slider dt_skew with a measured value. One-shot: separate from the
	// drift walk and from the continuous auto-tuner.
	struct LatencySample { double t; Eigen::Quaterniond qRef, qTgt; };
	bool   slamFixLatencyActive = false;
	double slamFixLatencyStartTime = 0.0;
	float  slamFixLatencyDurationS = 6.0f;
	std::vector<LatencySample> slamFixLatencyBuf;
	double slamFixLatencyLastMs = -1.0;   // last measured skew (ms), -1 = none this session
	double slamFixLatencyLastConf = 0.0;  // last measured confidence [0,1]

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
		slamFixScale = Eigen::Vector3d::Ones();
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
		slamFixScale = Eigen::Vector3d::Ones();
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
void StartSlamDriftCalibration();   // begin the timed manual drift-rate walk
void StartSlamLatencyCalibration(); // begin the one-time latency (time-skew) head-shake
void SlamFixApplyTuning();          // push CalCtx drift rates / time skew into the live filter
void LoadChaperoneBounds();
void ApplyChaperoneBounds();

void PushCalibrationApplyTime();
void ShowCalibrationDebug(int r, int c);
void DebugApplyRandomOffset();