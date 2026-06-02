#pragma once

#include <deque>
#include <utility>
#include <Eigen/Dense>

namespace Metrics {
	extern double TimeSpan, CurrentTime;

	double timestamp();
	void RecordTimestamp();

	template<typename T>
	class TimeSeries {
		std::deque<std::pair<double, T>> Data;
		
	public:
		const std::deque<std::pair<double, T>> &data() const { return Data; }

		void Push(const T& data) {
			Data.push_back(std::make_pair(CurrentTime, data));

			double cutoff = CurrentTime - TimeSpan;
			while (!Data.empty() && (Data.front().first < cutoff || Data.size() > INT_MAX)) {
				Data.pop_front();
			}
		}

		int size() const { return (int)Data.size(); }
		const std::pair<double, T>& operator[](int index) const { return Data[index]; }

		const T& last() const {
			static const T fallback;
			return Data.size() > 0 ? Data.back().second : fallback;
		}

		const double lastTs() const {
			return Data.size() > 0 ? Data.back().first : 0;
		}
	};


	extern TimeSeries<Eigen::Vector3d> posOffset_rawComputed; // , rotOffset_rawComputed;
	extern TimeSeries<Eigen::Vector3d> posOffset_currentCal; // , rotOffset_currentCal;
	extern TimeSeries<Eigen::Vector3d> posOffset_lastSample; // , rotOffset_lastSample;
	extern TimeSeries<Eigen::Vector3d> posOffset_byRelPose;
	
	extern TimeSeries<double> error_rawComputed, error_currentCal, error_byRelPose, error_currentCalRelPose;
	extern TimeSeries<double> axisIndependence;
	extern TimeSeries<double> computationTime;
	extern TimeSeries<double> jitterRef, jitterTarget;

	extern TimeSeries<bool> calibrationApplied;

	// SLAM-Fix per-frame metrics (only written in SLAM-Fix mode)
	// slamfix_phase: 0=bootstrap, 1=tracking, 2=reset (sustained Mahalanobis trip)
	extern TimeSeries<int> slamfix_phase;
	extern TimeSeries<double> slamfix_innov_pos_mm;
	extern TimeSeries<double> slamfix_innov_rot_deg;
	extern TimeSeries<double> slamfix_v_lin_mm_s;
	extern TimeSeries<double> slamfix_v_ang_deg_s;
	extern TimeSeries<double> slamfix_mahal;

	extern bool enableLogs;

	void WriteLogAnnotation(const char* s);
	void WriteLogEntry();

	// Returns true exactly once after a new log file is opened, then clears.
	// Used to re-emit the settings snapshot at the top of every fresh log.
	bool TakeLogOpenedFlag();
}