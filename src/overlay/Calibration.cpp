#include "stdafx.h"
#include "Calibration.h"
#include "CalibrationMetrics.h"
#include "Configuration.h"
#include "IPCClient.h"
#include "CalibrationCalc.h"
#include "VRState.h"

#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include <Eigen/Dense>
#include <GLFW/glfw3.h>

inline vr::HmdQuaternion_t operator*(const vr::HmdQuaternion_t& lhs, const vr::HmdQuaternion_t& rhs) {
	return {
		(lhs.w * rhs.w) - (lhs.x * rhs.x) - (lhs.y * rhs.y) - (lhs.z * rhs.z),
		(lhs.w * rhs.x) + (lhs.x * rhs.w) + (lhs.y * rhs.z) - (lhs.z * rhs.y),
		(lhs.w * rhs.y) + (lhs.y * rhs.w) + (lhs.z * rhs.x) - (lhs.x * rhs.z),
		(lhs.w * rhs.z) + (lhs.z * rhs.w) + (lhs.x * rhs.y) - (lhs.y * rhs.x)
	};
}

CalibrationContext CalCtx;
IPCClient Driver;
static protocol::DriverPoseShmem shmem;

namespace {
	CalibrationCalc calibration;

	inline vr::HmdVector3d_t quaternionRotateVector(const vr::HmdQuaternion_t& quat, const double(&vector)[3]) {
		vr::HmdQuaternion_t vectorQuat = { 0.0, vector[0], vector[1] , vector[2] };
		vr::HmdQuaternion_t conjugate = { quat.w, -quat.x, -quat.y, -quat.z };
		auto rotatedVectorQuat = quat * vectorQuat * conjugate;
		return { rotatedVectorQuat.x, rotatedVectorQuat.y, rotatedVectorQuat.z };
	}

	inline Eigen::Matrix3d quaternionRotateMatrix(const vr::HmdQuaternion_t& quat) {
		return Eigen::Quaterniond(quat.w, quat.x, quat.y, quat.z).toRotationMatrix();
	}

	struct DSample
	{
		bool valid;
		Eigen::Vector3d ref, target;
	};

	bool StartsWith(const std::string& str, const std::string& prefix)
	{
		if (str.length() < prefix.length())
			return false;

		return str.compare(0, prefix.length(), prefix) == 0;
	}

	bool EndsWith(const std::string& str, const std::string& suffix)
	{
		if (str.length() < suffix.length())
			return false;

		return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
	}

	Eigen::Vector3d AxisFromRotationMatrix3(Eigen::Matrix3d rot)
	{
		return Eigen::Vector3d(rot(2, 1) - rot(1, 2), rot(0, 2) - rot(2, 0), rot(1, 0) - rot(0, 1));
	}

	double AngleFromRotationMatrix3(Eigen::Matrix3d rot)
	{
		return acos((rot(0, 0) + rot(1, 1) + rot(2, 2) - 1.0) / 2.0);
	}

	vr::HmdQuaternion_t VRRotationQuat(const Eigen::Quaterniond& rotQuat)
	{

		vr::HmdQuaternion_t vrRotQuat;
		vrRotQuat.x = rotQuat.coeffs()[0];
		vrRotQuat.y = rotQuat.coeffs()[1];
		vrRotQuat.z = rotQuat.coeffs()[2];
		vrRotQuat.w = rotQuat.coeffs()[3];
		return vrRotQuat;
	}
	
	vr::HmdQuaternion_t VRRotationQuat(Eigen::Vector3d eulerdeg)
	{
		auto euler = eulerdeg * EIGEN_PI / 180.0;

		Eigen::Quaterniond rotQuat =
			Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitZ()) *
			Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY()) *
			Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitX());

		return VRRotationQuat(rotQuat);
	}

	vr::HmdVector3d_t VRTranslationVec(Eigen::Vector3d transcm)
	{
		auto trans = transcm * 0.01;
		vr::HmdVector3d_t vrTrans;
		vrTrans.v[0] = trans[0];
		vrTrans.v[1] = trans[1];
		vrTrans.v[2] = trans[2];
		return vrTrans;
	}

	DSample DeltaRotationSamples(Sample s1, Sample s2)
	{
		// Difference in rotation between samples.
		auto dref = s1.ref.rot * s2.ref.rot.transpose();
		auto dtarget = s1.target.rot * s2.target.rot.transpose();

		// When stuck together, the two tracked objects rotate as a pair,
		// therefore their axes of rotation must be equal between any given pair of samples.
		DSample ds;
		ds.ref = AxisFromRotationMatrix3(dref);
		ds.target = AxisFromRotationMatrix3(dtarget);

		// Reject samples that were too close to each other.
		auto refA = AngleFromRotationMatrix3(dref);
		auto targetA = AngleFromRotationMatrix3(dtarget);
		ds.valid = refA > 0.4 && targetA > 0.4 && ds.ref.norm() > 0.01 && ds.target.norm() > 0.01;

		ds.ref.normalize();
		ds.target.normalize();
		return ds;
	}

	Pose ConvertPose(const vr::DriverPose_t &driverPose) {
		Eigen::Quaterniond driverToWorldQ(
			driverPose.qWorldFromDriverRotation.w,
			driverPose.qWorldFromDriverRotation.x,
			driverPose.qWorldFromDriverRotation.y,
			driverPose.qWorldFromDriverRotation.z
		);
		Eigen::Vector3d driverToWorldV(
			driverPose.vecWorldFromDriverTranslation[0],
			driverPose.vecWorldFromDriverTranslation[1],
			driverPose.vecWorldFromDriverTranslation[2]
		);

		Eigen::Quaterniond driverRot = driverToWorldQ * Eigen::Quaterniond(
			driverPose.qRotation.w,
			driverPose.qRotation.x,
			driverPose.qRotation.y,
			driverPose.qRotation.z
		);
		
		Eigen::Vector3d driverPos = driverToWorldV + driverToWorldQ * Eigen::Vector3d(
			driverPose.vecPosition[0],
			driverPose.vecPosition[1],
			driverPose.vecPosition[2]
		);

		Eigen::AffineCompact3d xform = Eigen::Translation3d(driverPos) * driverRot;

		return Pose(xform);
	}

	bool CollectSample(const CalibrationContext& ctx)
	{
		vr::DriverPose_t reference, target;
		reference.poseIsValid = false;
		reference.result = vr::ETrackingResult::TrackingResult_Uninitialized;
		target.poseIsValid = false;
		target.result = vr::ETrackingResult::TrackingResult_Uninitialized;

		reference = ctx.devicePoses[ctx.referenceID];
		target = ctx.devicePoses[ctx.targetID];

		bool ok = true;
		if (!reference.poseIsValid && reference.result != vr::ETrackingResult::TrackingResult_Running_OK)
		{
			CalCtx.Log("Reference device is not tracking\n"); ok = false;
		}
		if (!target.poseIsValid && target.result != vr::ETrackingResult::TrackingResult_Running_OK)
		{
			CalCtx.Log("Target device is not tracking\n"); ok = false;
		}
		if (!ok)
		{
			if (CalCtx.state != CalibrationState::Continuous) {
				CalCtx.Log("Aborting calibration!\n");
				CalCtx.state = CalibrationState::None;
			}
			return false;
		}

		// Apply tracker offsets
		if (CalCtx.state == CalibrationState::Continuous || CalCtx.state == CalibrationState::ContinuousStandby) {
			reference.vecPosition[0] += ctx.continuousCalibrationOffset.x();
			reference.vecPosition[1] += ctx.continuousCalibrationOffset.y();
			reference.vecPosition[2] += ctx.continuousCalibrationOffset.z();
		}

		calibration.PushSample(Sample(
			ConvertPose(reference),
			ConvertPose(target),
			glfwGetTime()
		));

		return true;
	}

	bool AssignTargets() {
		auto state = VRState::Load();
		
		if (CalCtx.referenceID < 0) {
			CalCtx.referenceID = state.FindDevice(CalCtx.referenceStandby.trackingSystem, CalCtx.referenceStandby.model, CalCtx.referenceStandby.serial);
		}

		if (CalCtx.targetID < 0) {
			CalCtx.targetID = state.FindDevice(CalCtx.targetStandby.trackingSystem, CalCtx.targetStandby.model, CalCtx.targetStandby.serial);
		}

		for (int i = 0; i < CalCtx.MAX_CONTROLLERS; i++) {
			if (i < state.devices.size()
				&& state.devices[i].trackingSystem == CalCtx.targetTrackingSystem
				&& state.devices[i].deviceClass == vr::TrackedDeviceClass_Controller
				&& (state.devices[i].controllerRole == vr::TrackedControllerRole_LeftHand || state.devices[i].controllerRole == vr::TrackedControllerRole_RightHand))
			{
				CalCtx.controllerIDs[i] = state.devices[i].id;
			} else {
				CalCtx.controllerIDs[i] = -1;
			}
		}

		return CalCtx.referenceID >= 0 && CalCtx.targetID >= 0;
	}
}

void InitCalibrator()
{
	if (!Driver.TryConnect(5, 200))
	{
		std::cerr << "Failed to connect to Space Calibrator driver after retries, "
			<< "will retry on next calibration tick" << std::endl;
	}
	shmem.Open(OPENVR_SPACECALIBRATOR_SHMEM_NAME);
}

void ResetAndDisableOffsets(uint32_t id)
{
	vr::HmdVector3d_t zeroV;
	zeroV.v[0] = zeroV.v[1] = zeroV.v[2] = 0;

	vr::HmdQuaternion_t zeroQ;
	zeroQ.x = 0; zeroQ.y = 0; zeroQ.z = 0; zeroQ.w = 1;

	protocol::Request req(protocol::RequestSetDeviceTransform);
	req.setDeviceTransform = { id, false, zeroV, zeroQ, 1.0 };
	Driver.SendBlocking(req);
}

static_assert(vr::k_unTrackedDeviceIndex_Hmd == 0, "HMD index expected to be 0");

void ScanAndApplyProfile(CalibrationContext &ctx)
{
	std::unique_ptr<char[]> buffer_array(new char [vr::k_unMaxPropertyStringSize]);
	char* buffer = buffer_array.get();
	ctx.enabled = ctx.validProfile;

	protocol::Request setParamsReq(protocol::RequestSetAlignmentSpeedParams);
	setParamsReq.setAlignmentSpeedParams = ctx.EffectiveAlignmentSpeedParams();
	Driver.SendBlocking(setParamsReq);

	for (uint32_t id = 0; id < vr::k_unMaxTrackedDeviceCount; ++id)
	{
		auto deviceClass = vr::VRSystem()->GetTrackedDeviceClass(id);
		if (deviceClass == vr::TrackedDeviceClass_Invalid)
			continue;

		/*if (deviceClass == vr::TrackedDeviceClass_HMD) // for debugging unexpected universe switches
		{
			vr::ETrackedPropertyError err = vr::TrackedProp_Success;
			auto universeId = vr::VRSystem()->GetUint64TrackedDeviceProperty(id, vr::Prop_CurrentUniverseId_Uint64, &err);
			printf("uid %d err %d\n", universeId, err);
			ResetAndDisableOffsets(id);
			continue;
		}*/

		if (!ctx.enabled)
		{
			ResetAndDisableOffsets(id);
			continue;
		}

		vr::ETrackedPropertyError err = vr::TrackedProp_Success;
		vr::VRSystem()->GetStringTrackedDeviceProperty(id, vr::Prop_TrackingSystemName_String, buffer, vr::k_unMaxPropertyStringSize, &err);

		if (err != vr::TrackedProp_Success)
		{
			ResetAndDisableOffsets(id);
			continue;
		}

		std::string trackingSystem(buffer);

		if (id == vr::k_unTrackedDeviceIndex_Hmd)
		{
			//auto p = ctx.devicePoses[id].mDeviceToAbsoluteTracking.m;
			//printf("HMD %d: %f %f %f\n", id, p[0][3], p[1][3], p[2][3]);

			// Check if the current HMD is a Pimax crystal
			if (trackingSystem == "aapvr") {
				// HMD is a Pimax HMD
				vr::HmdMatrix34_t eyeToHeadLeft = vr::VRSystem()->GetEyeToHeadTransform(vr::Eye_Left);
				// Crystal's projection matrix is constant 0s or 1s except for [0][3], which stores the IPD offset from the nose
				bool isCrystalHmd =
					eyeToHeadLeft.m[0][0] == 1 && eyeToHeadLeft.m[0][1] == 0 && eyeToHeadLeft.m[0][2] == 0 &&                     // IPD
					eyeToHeadLeft.m[1][0] == 0 && eyeToHeadLeft.m[1][1] == 1 && eyeToHeadLeft.m[1][2] == 0 && eyeToHeadLeft.m[1][3] == 0 &&
					eyeToHeadLeft.m[2][0] == 0 && eyeToHeadLeft.m[2][1] == 0 && eyeToHeadLeft.m[2][2] == 1 && eyeToHeadLeft.m[2][3] == 0;

				if (isCrystalHmd) {
					// Move it outside the aapvr system ; we treat aapvr as if it were lighthouse
					trackingSystem = "Pimax Crystal HMD";
				}
			}

			if (trackingSystem != ctx.referenceTrackingSystem)
			{
				// Currently using an HMD with a different tracking system than the calibration.
				ctx.enabled = false;
			}

			ResetAndDisableOffsets(id);
			continue;
		}

		// Detect Pimax crystal controllers and separate them too
		if (deviceClass == vr::TrackedDeviceClass_Controller) {
			if (trackingSystem == "oculus") {
				vr::VRSystem()->GetStringTrackedDeviceProperty(id, vr::Prop_RenderModelName_String, buffer, vr::k_unMaxPropertyStringSize, &err);
				std::string renderModel(buffer);
				vr::VRSystem()->GetStringTrackedDeviceProperty(id, vr::Prop_ConnectedWirelessDongle_String, buffer, vr::k_unMaxPropertyStringSize, &err);
				std::string connectedWirelessDongle(buffer);

				// Check if the controller claims its an oculus controller but also pimax
				if (renderModel.find("{aapvr}") != std::string::npos &&
					renderModel.find("crystal") != std::string::npos &&
					connectedWirelessDongle.find("lighthouse") != std::string::npos) {
					trackingSystem = "Pimax Crystal Controllers";
				}
			}
		}

		if (trackingSystem != ctx.targetTrackingSystem)
		{
			ResetAndDisableOffsets(id);
			continue;
		}

		protocol::Request req(protocol::RequestSetDeviceTransform);
		req.setDeviceTransform = {
			id,
			true,
			VRTranslationVec(ctx.calibratedTranslation),
			VRRotationQuat(ctx.calibratedRotation),
			ctx.calibratedScale
		};
		req.setDeviceTransform.lerp = CalCtx.state == CalibrationState::Continuous;
		req.setDeviceTransform.quash = CalCtx.state == CalibrationState::Continuous && id == CalCtx.targetID && CalCtx.quashTargetInContinuous;

		Driver.SendBlocking(req);
	}

	if (ctx.enabled && ctx.chaperone.valid && ctx.chaperone.autoApply)
	{
		uint32_t quadCount = 0;
		vr::VRChaperoneSetup()->GetLiveCollisionBoundsInfo(nullptr, &quadCount);

		// Heuristic: when SteamVR resets to a blank-ish chaperone, it uses empty geometry,
		// but manual adjustments (e.g. via a play space mover) will not touch geometry.
		if (quadCount != ctx.chaperone.geometry.size())
		{
			ApplyChaperoneBounds();
		}
	}
}

void StartCalibration() {
	CalCtx.hasAppliedCalibrationResult = false;
	AssignTargets();
	CalCtx.state = CalibrationState::Begin;
	CalCtx.wantedUpdateInterval = 0.0;
	CalCtx.messages.clear();
	calibration.Clear();
	Metrics::WriteLogAnnotation("StartCalibration");
}

void StartContinuousCalibration() {
	CalCtx.hasAppliedCalibrationResult = false;
	AssignTargets();

	if (CalCtx.IsSlamFix()) {
		// Reset per-session SLAM-Fix state.
		CalCtx.slamFixTrackingTicks = 0;
		CalCtx.slamFixLastTickTime = 0.0;
		CalCtx.slamFixHmdPrevValid = false;
		CalCtx.slamFixVelLinEma = 0.0;
		CalCtx.slamFixVelAngEma = 0.0;
		CalCtx.slamFixRMountRefineTicks = 0;
		CalCtx.slamFixKabschRecenterTicks = 0;
		calibration.SlamFixDriftReset();
		// SLAM-Fix params are applied at-read-time via Effective*() accessors.
		// User profile fields are NOT mutated - switching back to FAST/SLOW/etc
		// restores the user's saved values.
		if (CalCtx.slamFixRLocked && CalCtx.relativePosCalibrated) {
			CalCtx.Log("SLAM-Fix mode: R loaded from profile, skipping bootstrap\n");
		} else {
			CalCtx.Log("SLAM-Fix mode: bootstrap (Kabsch) -> tracking (SE(3) EKF drift filter)\n");
		}
		CalCtx.Log("Settings locked: alignment speeds, thresholds, static recal, ignore outliers\n");
	}

	StartCalibration();
	CalCtx.state = CalibrationState::Continuous;
	calibration.setRelativeTransformation(CalCtx.refToTargetPose, CalCtx.relativePosCalibrated);
	calibration.lockRelativePosition = CalCtx.EffectiveLockRelativePosition();
	if (calibration.lockRelativePosition) {
		CalCtx.Log("Relative position locked");
	}
	else {
		CalCtx.Log("Collecting initial samples...");
	}
	Metrics::WriteLogAnnotation("StartContinuousCalibration");
}

void EndContinuousCalibration() {
	CalCtx.state = CalibrationState::None;
	CalCtx.relativePosCalibrated = false;
	SaveProfile(CalCtx);
	Metrics::WriteLogAnnotation("EndContinuousCalibration");
}

static const char* SpeedName(CalibrationContext::Speed s) {
	switch (s) {
		case CalibrationContext::SLAM_FIX:  return "SLAM_FIX";
		case CalibrationContext::FAST:      return "FAST";
		case CalibrationContext::SLOW:      return "SLOW";
		case CalibrationContext::VERY_SLOW: return "VERY_SLOW";
	}
	return "?";
}

static const char* StateName(CalibrationState s) {
	switch (s) {
		case CalibrationState::None:              return "None";
		case CalibrationState::Begin:             return "Begin";
		case CalibrationState::Rotation:          return "Rotation";
		case CalibrationState::Translation:       return "Translation";
		case CalibrationState::Editing:           return "Editing";
		case CalibrationState::Continuous:        return "Continuous";
		case CalibrationState::ContinuousStandby: return "ContinuousStandby";
	}
	return "?";
}

// Writes a "# [ts] SETTINGS ..." annotation into the debug log whenever the
// user-facing settings change (mode switch test<->SLAM_FIX, scale, thresholds,
// flags, devices) or when a fresh log file is opened. Called once per tick before
// WriteLogEntry; change-detection keeps it from repeating every frame.
static void LogSettingsIfChanged() {
	if (!Metrics::enableLogs) return;

	const auto& c = CalCtx;
	const auto asp = c.EffectiveAlignmentSpeedParams();
	std::ostringstream ss;
	ss << "SETTINGS"
	   << " mode=" << SpeedName(c.calibrationSpeed) << (c.IsSlamFix() ? "(slam)" : "(test)")
	   << " state=" << StateName(c.state)
	   << " scale=" << c.calibratedScale
	   << " refID=" << c.referenceID << " targetID=" << c.targetID
	   << " refSys=" << (c.referenceTrackingSystem.empty() ? "-" : c.referenceTrackingSystem)
	   << " tgtSys=" << (c.targetTrackingSystem.empty() ? "-" : c.targetTrackingSystem)
	   << " contThr=" << c.EffectiveContinuousCalibrationThreshold()
	   << " maxRelErr=" << c.EffectiveMaxRelativeErrorThreshold()
	   << " jitter=" << c.EffectiveJitterThreshold()
	   << " staticRecal=" << (c.EffectiveStaticRecalibration() ? 1 : 0)
	   << " lockRelPos=" << (c.EffectiveLockRelativePosition() ? 1 : 0)
	   << " ignoreOutliers=" << (c.EffectiveIgnoreOutliers() ? 1 : 0)
	   << " quashInCont=" << (c.quashTargetInContinuous ? 1 : 0)
	   << " slamFixRLocked=" << (c.slamFixRLocked ? 1 : 0)
	   << " enabled=" << (c.enabled ? 1 : 0)
	   << " validProfile=" << (c.validProfile ? 1 : 0)
	   << " alignSpd[t/s/l]=" << asp.align_speed_tiny << "/" << asp.align_speed_small << "/" << asp.align_speed_large;

	std::string snap = ss.str();
	static std::string lastSnap;
	bool reopened = Metrics::TakeLogOpenedFlag();
	if (reopened || snap != lastSnap) {
		Metrics::WriteLogAnnotation(snap.c_str());
		Metrics::TakeLogOpenedFlag(); // consume the flag set if this write opened the file
		lastSnap = snap;
	}
}

void CalibrationTick(double time)
{
	if (!vr::VRSystem())
		return;

	auto &ctx = CalCtx;
	double tickInterval = ctx.IsSlamFix() ? 0.01 : 0.05;
	if ((time - ctx.timeLastTick) < tickInterval)
		return;

	if (ctx.state == CalibrationState::Continuous || ctx.state == CalibrationState::ContinuousStandby) {
		ctx.ClearLogOnMessage();

		if (CalCtx.requireTriggerPressToApply && (time - ctx.timeLastAssign) > 10) {
			// rescan devices every 10 seconds or so if we are using controller data
			ctx.timeLastAssign = time;
			AssignTargets();
		}
	}

	ctx.timeLastTick = time;
	shmem.ReadNewPoses([&](const protocol::DriverPoseShmem::AugmentedPose& augmented_pose) {
		if (augmented_pose.deviceId >= 0 && augmented_pose.deviceId <= vr::k_unMaxTrackedDeviceCount) {
			ctx.devicePoses[augmented_pose.deviceId] = augmented_pose.pose;
		}
	});

	// check for non-updating headset tracking space (caused by quest out of bounds or taken off head for example) and abort everything for this tick
	auto p = ctx.devicePoses[vr::k_unTrackedDeviceIndex_Hmd].vecPosition;
	if ((p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0) || (ctx.xprev == p[0] && ctx.yprev == p[1] && ctx.zprev == p[2])) {
		// std::cerr << "HMD tracking didn't update, skipping update" << std::endl;
		return;
	}
	ctx.xprev = (float) p[0];
	ctx.yprev = (float) p[1];
	ctx.zprev = (float) p[2];

	if (ctx.state == CalibrationState::None || ctx.state == CalibrationState::ContinuousStandby
		|| (ctx.state == CalibrationState::Continuous && !calibration.isValid()))
	{
		if ((time - ctx.timeLastScan) >= 1.0)
		{
			ScanAndApplyProfile(ctx);
			ctx.timeLastScan = time;
		}
	}

	if (ctx.state == CalibrationState::ContinuousStandby) {
		if (AssignTargets()) {
			StartContinuousCalibration();
		}
		else {
			ctx.wantedUpdateInterval = 0.5;
			ctx.Log("Waiting for devices...\n");
			return;
		}
	}

	if (ctx.state == CalibrationState::None) {
		ctx.wantedUpdateInterval = 1.0;
		return;
	}

	if (ctx.state == CalibrationState::Editing)
	{
		ctx.wantedUpdateInterval = 0.1;

		if ((time - ctx.timeLastScan) >= 0.1)
		{
			ScanAndApplyProfile(ctx);
			ctx.timeLastScan = time;
		}
		return;
	}

	bool ok = true;

	if (ctx.referenceID == -1 || ctx.referenceID >= vr::k_unMaxTrackedDeviceCount) {
		CalCtx.Log("Missing reference device\n");
		ok = false;
	}
	if (ctx.targetID == -1 || ctx.targetID >= vr::k_unMaxTrackedDeviceCount)
	{
		CalCtx.Log("Missing target device\n");
		ok = false;
	}

	if (ctx.state == CalibrationState::Begin)
	{

		char referenceSerial[256], targetSerial[256];
		referenceSerial[0] = targetSerial[0] = 0;
		vr::VRSystem()->GetStringTrackedDeviceProperty(ctx.referenceID, vr::Prop_SerialNumber_String, referenceSerial, 256);
		vr::VRSystem()->GetStringTrackedDeviceProperty(ctx.targetID, vr::Prop_SerialNumber_String, targetSerial, 256);

		char buf[256];
		snprintf(buf, sizeof buf, "Reference device ID: %d, serial: %s\n", ctx.referenceID, referenceSerial);
		CalCtx.Log(buf);
		snprintf(buf, sizeof buf, "Target device ID: %d, serial %s\n", ctx.targetID, targetSerial);
		CalCtx.Log(buf);

		ScanAndApplyProfile(ctx);

		Metrics::jitterRef.Push(calibration.ReferenceJitter());
		Metrics::jitterRef.Push(calibration.TargetJitter());

		if (!CalCtx.ReferencePoseIsValidSimple())
		{
			CalCtx.Log("Reference device is not tracking\n"); ok = false;
		}

		if (!CalCtx.TargetPoseIsValidSimple())
		{
			CalCtx.Log("Target device is not tracking\n"); ok = false;
		}
		
		// @TOOD: Determine if the tracking is jittery
		if (calibration.ReferenceJitter() > ctx.EffectiveJitterThreshold()) {
			CalCtx.Log("Reference device is not tracking\n"); ok = false;
		}
		if (calibration.TargetJitter() > ctx.EffectiveJitterThreshold()) {
			CalCtx.Log("Target device is not tracking\n"); ok = false;
		}

		if (ok) {
			//ResetAndDisableOffsets(ctx.targetID);
			ctx.state = CalibrationState::Rotation;
			ctx.wantedUpdateInterval = 0.0;

			CalCtx.Log("Starting calibration...\n");
			return;
		}
	}

	if (!ok)
	{
		if (ctx.state != CalibrationState::Continuous) {
			ctx.state = CalibrationState::None;

			CalCtx.Log("Aborting calibration!\n");
		}
		return;
	}

	if (!CollectSample(ctx))
	{
		return;
	}

	// SLAM-Fix tracking phase: per-frame SE(3) EKF drift filter.
	// Activates only after bootstrap has locked the rigid offset R
	// (calibration.isRelativeTransformationCalibrated() == true).
	// Bypasses the Kabsch sliding-window path for in-tracking updates.
	//
	// The EKF replaces the old LPF + binary motion gate + jump-streak design.
	// - Process noise Q scales with HMD speed (filter trusts SLAM less while moving)
	// - Measurement noise R inflates with (omega * lever_arm)^2 (no sample dropping)
	// - Sustained Mahalanobis > thresh -> snap to T_meas + inflate covariance
	if (CalCtx.state == CalibrationState::Continuous
		&& CalCtx.IsSlamFix()
		&& calibration.isRelativeTransformationCalibrated()
		&& calibration.SampleCount() >= 1)
	{
		// dt since previous SLAM-Fix tick (clamped inside SlamFixDriftStep).
		double dt = (ctx.slamFixLastTickTime > 0.0) ? (time - ctx.slamFixLastTickTime) : 0.01;
		ctx.slamFixLastTickTime = time;

		// FAST→SLAM gap detection: if dt > 1s, the user switched away from
		// SLAM mode (e.g. to FAST) and back. During the gap, FAST may have
		// corrected the calibration, but the EKF still holds stale state from
		// before the switch. Reset EKF from the current T_meas so it picks up
		// whatever FAST computed, instead of snapping back to the old offset.
		if (dt > 1.0 && calibration.isValid()) {
			char gapBuf[128];
			snprintf(gapBuf, sizeof gapBuf, "SLAM-Fix: gap detected (dt=%.1fs), resetting EKF from T_meas\n", dt);
			CalCtx.Log(gapBuf);
			calibration.SlamFixDriftReset();
			// Force re-bootstrap on next tick (m_isValid stays true but filter
			// is re-initialized, so the next SlamFixDriftStep will snap to T_meas).
		}

		// HMD velocity magnitudes for motion-correlated Q and lever-arm R inflation.
		// Many SLAM HMDs (Pico/Quest streaming via VirtualDesktop/ALVR) leave
		// DriverPose_t::vecVelocity at zero, so we always finite-difference
		// from successive HMD poses. The driver-reported velocity is used
		// only as a sanity floor (max of the two).
		const auto& hmdPose = ctx.devicePoses[vr::k_unTrackedDeviceIndex_Hmd];
		double driver_lin_speed = std::sqrt(
			hmdPose.vecVelocity[0]*hmdPose.vecVelocity[0] +
			hmdPose.vecVelocity[1]*hmdPose.vecVelocity[1] +
			hmdPose.vecVelocity[2]*hmdPose.vecVelocity[2]);
		double driver_ang_speed = std::sqrt(
			hmdPose.vecAngularVelocity[0]*hmdPose.vecAngularVelocity[0] +
			hmdPose.vecAngularVelocity[1]*hmdPose.vecAngularVelocity[1] +
			hmdPose.vecAngularVelocity[2]*hmdPose.vecAngularVelocity[2]);

		double fd_lin_speed = 0.0, fd_ang_speed = 0.0;
		if (ctx.slamFixHmdPrevValid && dt > 1e-4) {
			double dx = hmdPose.vecPosition[0] - ctx.slamFixHmdPrevX;
			double dy = hmdPose.vecPosition[1] - ctx.slamFixHmdPrevY;
			double dz = hmdPose.vecPosition[2] - ctx.slamFixHmdPrevZ;
			fd_lin_speed = std::sqrt(dx*dx + dy*dy + dz*dz) / dt;

			Eigen::Quaterniond qPrev(ctx.slamFixHmdPrevQw, ctx.slamFixHmdPrevQx,
			                         ctx.slamFixHmdPrevQy, ctx.slamFixHmdPrevQz);
			Eigen::Quaterniond qNow(hmdPose.qRotation.w, hmdPose.qRotation.x,
			                        hmdPose.qRotation.y, hmdPose.qRotation.z);
			qPrev.normalize(); qNow.normalize();
			if (qPrev.dot(qNow) < 0.0) qNow.coeffs() = -qNow.coeffs();
			Eigen::Quaterniond qDelta = qPrev.conjugate() * qNow;
			qDelta.normalize();
			double dAngle = 2.0 * std::acos(std::min(1.0, std::abs(qDelta.w())));
			fd_ang_speed = dAngle / dt;
		}
		ctx.slamFixHmdPrevX = hmdPose.vecPosition[0];
		ctx.slamFixHmdPrevY = hmdPose.vecPosition[1];
		ctx.slamFixHmdPrevZ = hmdPose.vecPosition[2];
		ctx.slamFixHmdPrevQw = hmdPose.qRotation.w;
		ctx.slamFixHmdPrevQx = hmdPose.qRotation.x;
		ctx.slamFixHmdPrevQy = hmdPose.qRotation.y;
		ctx.slamFixHmdPrevQz = hmdPose.qRotation.z;
		ctx.slamFixHmdPrevValid = true;

		double user_lin_speed_raw = std::max(driver_lin_speed, fd_lin_speed);
		double user_ang_speed_raw = std::max(driver_ang_speed, fd_ang_speed);

		// Hard-clamp to plausibly-human speeds. SLAM teleports / pose buffer
		// hiccups produce 100x-1000x spikes (e.g. 245 m/s lin, 130 rad/s ang
		// were observed in real logs). Anything above these limits is not real
		// motion and must not pump process noise.
		const double V_LIN_MAX = 3.0;       // m/s - human running
		const double V_ANG_MAX = 6.0;       // rad/s ~= 343 deg/s - very fast head turn
		if (user_lin_speed_raw > V_LIN_MAX) user_lin_speed_raw = V_LIN_MAX;
		if (user_ang_speed_raw > V_ANG_MAX) user_ang_speed_raw = V_ANG_MAX;

		// EMA smoothing for Q only. alpha=0.3 -> ~3-tick (30ms) time constant.
		// Single outlier moves EMA by 30% then decays - filter sees a small
		// transient instead of a giant Q kick.
		const double VEL_EMA_ALPHA = 0.3;
		ctx.slamFixVelLinEma = (1.0 - VEL_EMA_ALPHA) * ctx.slamFixVelLinEma + VEL_EMA_ALPHA * user_lin_speed_raw;
		ctx.slamFixVelAngEma = (1.0 - VEL_EMA_ALPHA) * ctx.slamFixVelAngEma + VEL_EMA_ALPHA * user_ang_speed_raw;
		double user_lin_speed_q = ctx.slamFixVelLinEma;
		double user_ang_speed_q = ctx.slamFixVelAngEma;
		// R uses raw clamped values - inflation must engage immediately at
		// rotation onset, not lag by EMA tau (3 ticks of unprotected updates
		// at the start of every head turn caused chronic mis-rotation).
		double user_lin_speed_r = user_lin_speed_raw;
		double user_ang_speed_r = user_ang_speed_raw;

		// Lever arm: actual puck-to-HMD offset magnitude from the locked R_mount
		// (was a hardcoded 0.10). The R-inflation that freezes position during a
		// head turn must match the true geometry or rotation leaks into the
		// estimate. Clamped to a sane range in case R_mount is degenerate.
		double lever_arm_m = calibration.RelativeTransformation().translation().norm();
		if (lever_arm_m < 0.05) lever_arm_m = 0.05;
		if (lever_arm_m > 0.30) lever_arm_m = 0.30;

		double innov_pos = 0.0, innov_rot = 0.0, mahal = 0.0;
		bool ok_lp = calibration.SlamFixDriftStep(
			dt, user_lin_speed_q, user_ang_speed_q,
			user_lin_speed_r, user_ang_speed_r,
			lever_arm_m,
			&innov_pos, &innov_rot, &mahal);

		// Phase classification for log:
		//   0 = bootstrap (filter not yet initialized)
		//   1 = tracking (normal predict+update)
		//   2 = reset (sustained Mahalanobis trip just snapped state to T_meas)
		int slamfix_phase = 1;
		if (calibration.SlamFixConsumeResetEvent()) slamfix_phase = 2;

		if (ok_lp && calibration.isValid()) {
			ctx.calibratedRotation = calibration.EulerRotation();
			ctx.calibratedTranslation = calibration.Transformation().translation() * 100.0; // cm
			// Do NOT update refToTargetPose during tracking - R stays locked.
			ctx.validProfile = true;
			ScanAndApplyProfile(ctx);
			CalCtx.hasAppliedCalibrationResult = true;

			// Auto-save R to profile after 200 stable tracking ticks (~2s).
			// This lets subsequent sessions skip bootstrap entirely.
			ctx.slamFixTrackingTicks++;
			if (!ctx.slamFixRLocked && ctx.slamFixTrackingTicks >= 200) {
				ctx.slamFixRLocked = true;
				SaveProfile(ctx);
				CalCtx.Log("SLAM-Fix: R locked and saved to profile (bootstrap skipped next session)\n");
			}
		}

		// Write SLAM-Fix metrics for log analysis.
		Metrics::RecordTimestamp();
		Metrics::slamfix_phase.Push(slamfix_phase);
		Metrics::slamfix_innov_pos_mm.Push(innov_pos * 1000.0);
		Metrics::slamfix_innov_rot_deg.Push(innov_rot * 180.0 / EIGEN_PI);
		Metrics::slamfix_v_lin_mm_s.Push(user_lin_speed_r * 1000.0);
		Metrics::slamfix_v_ang_deg_s.Push(user_ang_speed_r * 180.0 / EIGEN_PI);
		Metrics::slamfix_mahal.Push(mahal);

		// RefineRMount REMOVED: Both implementations are harmful.
		// - EKF-based: circular lock (drift → R_mount → confirms drift)
		// - Kabsch-based: garbage results with low rotation variance (3485mm jumps)
		// R_mount stays frozen from bootstrap. Kabsch recenter below handles
		// periodic correction when there's enough rotational variance.

		// Kabsch recenter: confidence-gated, driven by FAST's acceptance test.
		// The 100-tick (~1s) cadence is only an evaluation budget (a full Kabsch
		// SVD is too costly per frame); whether it actually fires is decided
		// entirely inside SlamFixKabschRecenter by the SAME confidence gate the
		// FAST/continuous preset uses (variance + absolute error + improvement
		// over the current EKF state by the contThr margin). So it recenters only
		// when FAST itself would be certain - not on the timer, not in low motion.
		// Feeds axis variance to the debug graph so corrections are visible.
		ctx.slamFixKabschRecenterTicks++;
		if (ctx.slamFixKabschRecenterTicks >= 100
			&& calibration.SampleCount() >= 50)
		{
			ctx.slamFixKabschRecenterTicks = 0;
			double axVar = 0.0;
			if (calibration.SlamFixKabschRecenter(true,
					CalCtx.EffectiveContinuousCalibrationThreshold(),
					CalCtx.EffectiveMaxRelativeErrorThreshold(),
					&axVar)) {
				ctx.calibratedRotation = calibration.EulerRotation();
				ctx.calibratedTranslation = calibration.Transformation().translation() * 100.0;
				ctx.validProfile = true;
				ScanAndApplyProfile(ctx);
				CalCtx.Log("SLAM-Fix: Kabsch recenter corrected drift\n");
			}
			// Push axis variance to debug graph regardless of correction.
			Metrics::axisIndependence.Push(axVar);
		}

		// Write standard metrics for debug graphs (otherwise frozen/stale
		// during SLAM-Fix tracking because ComputeIncremental is bypassed).
		{
			double rmsError = 0.0;
			Eigen::Vector3d posOff;
			calibration.ComputeCurrentCalMetrics(&rmsError, &posOff);
			Metrics::error_currentCal.Push(rmsError * 1000.0);
			Metrics::posOffset_currentCal.Push(posOff * 1000.0);
		}

		// Trim the sample buffer - keep enough for Kabsch.
		while (calibration.SampleCount() > 200) calibration.ShiftSample();

		LogSettingsIfChanged();
		Metrics::WriteLogEntry();
		return;
	}

	CalCtx.Progress((int) calibration.SampleCount(), (int)CalCtx.SampleCount());

	if (calibration.SampleCount() < CalCtx.SampleCount()) return;
	while (calibration.SampleCount() > CalCtx.SampleCount()) calibration.ShiftSample();

	if (CalCtx.state == CalibrationState::Continuous && CalCtx.requireTriggerPressToApply && CalCtx.hasAppliedCalibrationResult) {
		bool triggerPressed = true;
		vr::VRControllerState_t state;
		for (int i = 0; i < CalCtx.MAX_CONTROLLERS; i++) {
			if (CalCtx.controllerIDs[i] >= 0) {
				vr::VRSystem()->GetControllerState(CalCtx.controllerIDs[i], &state, sizeof(state));
				triggerPressed &= state.rAxis[vr::k_eControllerAxis_TrackPad /* matches trigger on Index controllers?? */].x > 0.75f
					|| state.rAxis[vr::k_eControllerAxis_Trigger].x > 0.75f;
				//printf("Controller %d tracpad: %f\n", i, state.rAxis[vr::k_eControllerAxis_TrackPad].x);
				//printf("Controller %d trigger: %f\n", i, state.rAxis[vr::k_eControllerAxis_Trigger].x);
				if (!triggerPressed) {
					break;
				}
			}
		}

		if (!triggerPressed) {
			CalCtx.Log("Waiting for trigger press...\n");
			CalCtx.wasWaitingForTriggers = true;
			return;
		}

		if (CalCtx.wasWaitingForTriggers) {
			CalCtx.Log("Triggers pressed, continuing calibration...\n");
			CalCtx.wasWaitingForTriggers = false;
		}
	}

	LARGE_INTEGER start_time;
	QueryPerformanceCounter(&start_time);
		
	bool lerp = false;

	if (CalCtx.state == CalibrationState::Continuous) {
		CalCtx.messages.clear();
		calibration.enableStaticRecalibration = CalCtx.EffectiveStaticRecalibration();
		calibration.lockRelativePosition = CalCtx.EffectiveLockRelativePosition();
		calibration.ComputeIncremental(lerp,
			CalCtx.EffectiveContinuousCalibrationThreshold(),
			CalCtx.EffectiveMaxRelativeErrorThreshold(),
			CalCtx.EffectiveIgnoreOutliers());
	}
	else {
		calibration.enableStaticRecalibration = false;
		calibration.ComputeOneshot(CalCtx.EffectiveIgnoreOutliers());
	}

	if (calibration.isValid()) {
		ctx.calibratedRotation = calibration.EulerRotation();
		ctx.calibratedTranslation = calibration.Transformation().translation() * 100.0; // convert to cm units for profile storage
		ctx.refToTargetPose = calibration.RelativeTransformation();
		ctx.relativePosCalibrated = calibration.isRelativeTransformationCalibrated();

		auto vrTrans = VRTranslationVec(ctx.calibratedTranslation);
		auto vrRot = VRRotationQuat(Eigen::Quaterniond(calibration.Transformation().rotation()));

		ctx.validProfile = true;
		SaveProfile(ctx);

		ScanAndApplyProfile(ctx);

		CalCtx.hasAppliedCalibrationResult = true;

		CalCtx.Log("Finished calibration, profile saved\n");
	} else {
		CalCtx.Log("Calibration failed.\n");
	}

	LARGE_INTEGER end_time;
	QueryPerformanceCounter(&end_time);
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	double duration = (end_time.QuadPart - start_time.QuadPart) / (double)freq.QuadPart;
	Metrics::computationTime.Push(duration * 1000.0);

	// In bootstrap phase (SLAM-Fix or any other mode), phase=0.
	if (CalCtx.IsSlamFix()) {
		Metrics::slamfix_phase.Push(0);
		Metrics::slamfix_innov_pos_mm.Push(0.0);
		Metrics::slamfix_innov_rot_deg.Push(0.0);
	}

	LogSettingsIfChanged();
	Metrics::WriteLogEntry();
		
	if (CalCtx.state != CalibrationState::Continuous) {
		ctx.state = CalibrationState::None;
		calibration.Clear();
	}
	else {
		// SLAM-Fix bootstrap: drop 40% of samples per cycle for faster sliding window
		size_t drop_samples = CalCtx.IsSlamFix()
			? CalCtx.SampleCount() * 2 / 5
			: CalCtx.SampleCount() / 10;
		for (size_t i = 0; i < drop_samples; i++) {
			calibration.ShiftSample();
		}
	}
}

void LoadChaperoneBounds()
{
	vr::VRChaperoneSetup()->RevertWorkingCopy();

	uint32_t quadCount = 0;
	vr::VRChaperoneSetup()->GetLiveCollisionBoundsInfo(nullptr, &quadCount);

	CalCtx.chaperone.geometry.resize(quadCount);
	vr::VRChaperoneSetup()->GetLiveCollisionBoundsInfo(&CalCtx.chaperone.geometry[0], &quadCount);
	vr::VRChaperoneSetup()->GetWorkingStandingZeroPoseToRawTrackingPose(&CalCtx.chaperone.standingCenter);
	vr::VRChaperoneSetup()->GetWorkingPlayAreaSize(&CalCtx.chaperone.playSpaceSize.v[0], &CalCtx.chaperone.playSpaceSize.v[1]);
	CalCtx.chaperone.valid = true;
}

void ApplyChaperoneBounds()
{
	vr::VRChaperoneSetup()->RevertWorkingCopy();
	vr::VRChaperoneSetup()->SetWorkingCollisionBoundsInfo(&CalCtx.chaperone.geometry[0], (uint32_t)CalCtx.chaperone.geometry.size());
	vr::VRChaperoneSetup()->SetWorkingStandingZeroPoseToRawTrackingPose(&CalCtx.chaperone.standingCenter);
	vr::VRChaperoneSetup()->SetWorkingPlayAreaSize(CalCtx.chaperone.playSpaceSize.v[0], CalCtx.chaperone.playSpaceSize.v[1]);
	vr::VRChaperoneSetup()->CommitWorkingCopy(vr::EChaperoneConfigFile_Live);
}

void DebugApplyRandomOffset() {
	protocol::Request req(protocol::RequestDebugOffset);
	Driver.SendBlocking(req);
}