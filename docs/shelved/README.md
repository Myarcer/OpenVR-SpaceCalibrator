# Shelved: SLAM time-skew / latency calibration

Removed from the build (UI + code) because the measured reference↔SLAM time
skew is **not observable enough** under the motion humans actually make, and
every way of *consuming* it routed a velocity-correlated quantity into the
rigid calibration's rotation DOF — baking in a coherent **scene tilt** (pure-X
motion emitting Y/Z vectors). See `SLAM_TIME_ALIGNMENT.md` for the original
design, and the deep-research notes for the observability argument
(skew is only observable under high angular *acceleration*).

What was removed:

- `EstimateTimeSkew` cross-correlation + `StartSlamLatencyCalibration` head-shake
- `LatencySample` buffer / collection loop / `LatencyResult` logging
- Reference-history interpolation de-skew (`InterpolateRef` / `g_refHistory`)
- DriftFilter Phase-0 Exp-twist de-skew + `dt_skew_s` / `k_skew` R-inflation
  terms + the `hmd_*_vel_body` twist plumbing
- `slamFixTimeSkew` field, profile load/save, and the **Time skew (ms)** slider
  + **Calibrate latency (shake ~6s)** UI

What was kept: the SE(3) EKF drift filter, lever-arm + omega R-inflation,
ZUPT, and the (eager) Kabsch recenter. The directionless `k_skew` damping was
the only "safe" use; it's gone too since `dt_skew_s` is now always 0.

## Recovering the full implementation

The complete working code is preserved at the git tag **`shelved/slam-time-skew`**
(the commit immediately before removal):

```sh
git show shelved/slam-time-skew                       # browse
git checkout shelved/slam-time-skew -- src/overlay/   # restore files
```

If revisited, the only mathematically safe variant is a **linear-only** de-skew
(zero the angular twist — kills the SE(3) screw cross-coupling) gated on
angular *acceleration* confidence, not angular velocity.
