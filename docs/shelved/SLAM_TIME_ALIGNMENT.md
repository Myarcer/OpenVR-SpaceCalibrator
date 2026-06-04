# Real Timestamp-Alignment for SLAM ↔ Lighthouse

Design + implementation plan for consuming the measured `slam_fix_time_skew`
as a **temporal alignment** of the two tracking streams, instead of only as an
EKF measurement-noise inflation term.

Status: **Phase 0 done** (sign fix landed). Phases 1–4 planned below.

---

## 1. Background: what the time-skew is

SpaceCalibrator pairs a **reference** pose (SteamVR Lighthouse, low latency,
~100+ Hz, smooth) with a **target** pose (SLAM headset/controller streamed over
VirtualDesktop/ALVR — Pico/Quest — with ~10–25 ms transport latency).

`slam_fix_time_skew` (`dt_skew_s`) is the estimated transport delay between the
two streams, recovered by **normalized cross-correlation of the two streams'
angular-speed signals** (`EstimateTimeSkew`, `Calibration.cpp`). The sign is
defined as:

```
tgt(t) ≈ ref(t − skew)     skew > 0 ⇒ SLAM lags reference (normal streaming)
                            skew < 0 ⇒ SLAM leads (ALVR/VD pose overprediction)
```

This is `|omega|`-based because angular speed magnitude is invariant to the
rigid-body lever-arm offset and to the (not-yet-solved) mount frame, so it works
before calibration is perfect and needs no frame alignment.

## 2. The actual problem

Every paired sample is captured at **one wall-clock instant** in
`CollectSample` (`Calibration.cpp`):

```cpp
calibration.PushSample(Sample(
    ConvertPose(reference),   // lighthouse @ now
    ConvertPose(target),      // SLAM @ now, but PHYSICALLY ~now − skew
    glfwGetTime()             // a SINGLE timestamp for both
));
```

The two poses are read simultaneously but describe **different physical
instants**: the SLAM pose is physically `skew` seconds old. During motion every
pair is offset by ≈ `v·skew` (translation) and `ω·skew` (rotation).

Every downstream consumer treats `ref` and `target` as simultaneous:

| Consumer | Function | What it reads |
| --- | --- | --- |
| Outlier reject | `DetectOutliers` | ref/target delta-rotation pairs |
| Kabsch rotation | `CalibrateRotation` | ref/target delta-rotation axis pairs |
| Translation solve | `CalibrateTranslation` | `ref.trans − target.trans` per sample |
| Per-axis scale | `EstimatePerAxisScale` | ref vs target displacement |
| EKF measurement | `SlamFixDriftStep` → `T_meas` | `ref · R_mount · target⁻¹` |
| Mount refine | `RefineRMount` | stateless Kabsch over buffer |
| Kabsch recenter | `SlamFixKabschRecenter` | stateless Kabsch over buffer |

## 3. What was built before this work (the conservative fallback)

`dt_skew_s` only fed the EKF **R-inflation** (`DriftFilter.cpp`):

```cpp
double skew_disp = user_lin_speed_mps * params.dt_skew_s;   // m
double skew_rot  = user_ang_speed_radps * params.dt_skew_s; // rad
R_pos += k_skew     * skew_disp * skew_disp;
R_rot += k_skew_rot * skew_rot  * skew_rot;
```

This **squares** `dt_skew_s`, so the SIGN is discarded — `+18 ms` and `−18 ms`
behave identically. It does **not** delay correction application (a fear worth
ruling out — it does not), and it does **not** time-align the streams. It only
says "during motion these two streams are inherently `v·skew` apart, so distrust
the mismatch." Safe and simple, but throws away recoverable accuracy and ignores
the sign the cross-correlation went to the trouble of measuring.

> Note: variance is inherently a magnitude, so squaring in **R-inflation is
> mathematically correct there** — the sign genuinely cannot matter for a noise
> term. The sign only becomes meaningful for a **directional correction**
> (shifting the residual / re-timestamping a pose), which is the work below.

---

## 4. Research conclusions (deep-research 2026-06-04)

Source: `Research/deep-research/research_20260604_023344_*.md`. Grounded in
VINS-Mono online temporal calibration, Kalibr, Sola "Micro Lie Theory", Sophus.

1. **Interpolate the low-latency (lighthouse) stream BACKWARD** to the SLAM
   timestamp. Do NOT extrapolate SLAM forward — extrapolation integrates
   prediction error and overshoots on high-frequency head motion. The delayed
   sensor (SLAM) is the temporal anchor; the clean/fast stream is interpolated to
   it. This is what Kalibr / VINS-Mono do.

2. **First-order de-skew on the manifold, body-frame twist, right-multiply:**

   ```
   T_corrected = T_meas · Exp([v; ω] · dt)
   ```

   with `v`, `ω` in the **body frame** (Sophus right-multiplication convention,
   `T_wb = T_wb · Exp(ξ_b)`). World-frame is valid but would require rotating the
   velocities into world each tick — body-frame is the natural fit and matches
   how `DriftFilter::Predict` already does `T_ = T_ * Exp(v·dt)`.

3. **Decoupled SLERP (rotation) + LERP (translation)** for buffer interpolation —
   NOT Sophus geodesic `interpolate()`, which produces curved "screw-motion"
   translation. A user moving in a straight line while turning should get a
   straight-line translation.

4. **Dropped-frame gating:** if the bracketing buffer samples are more than
   ~3× the expected interval apart, do NOT interpolate — mark the sample invalid.
   Reuse the gap-rejection already in `EstimateTimeSkew` (`dt > 0.1` drop).

5. **Kabsch benefits** from alignment: uncorrected, the delta-rotation axis pairs
   are corrupted by the inter-stream phase shift, producing a "lazy" rotation fix
   that lags real motion.

6. **Order matters with anisotropic scale:** the `v` used in `v·dt` is itself
   wrong if SLAM scale is off (e.g. 1.05×). Apply scale correction *before*
   computing the de-skew velocity, or accept a second-order residual.

7. **Observability caveat:** `dt_skew` is only observable under **high angular
   acceleration**. Under static or constant-velocity motion the time offset is
   indistinguishable from a spatial offset — so the estimate must stay gated on
   correlation confidence and motion energy (already done).

8. **Signal filtering:** low-pass the `ω` signals (5–10 Hz) before NCC to avoid
   false correlation peaks from transport jitter producing a jumpy `dt`.

---

## 5. Architecture decision

**Align at the single source (`CollectSample`), not at six consumers.**

De-skew once, where the `Sample` is built, so every downstream consumer
(Kabsch, translation, per-axis scale, EKF `T_meas`, mount refine, recenter)
becomes correct automatically without touching their math. The existing
R-inflation stays as a *residual* safety net for interpolation error and
skew-estimate uncertainty `δskew`, rather than being the only mechanism.

---

## 6. Phased plan

### Phase 0 — Sign fix (DONE)

Smallest real consumption of the sign, no buffer required. Plumb the **signed**
HMD velocity twist (already available at the call site as
`hmdPose.vecVelocity[0..2]` / `vecAngularVelocity[0..2]`, currently collapsed to
scalar magnitudes) into the EKF update, and apply a first-order directional
de-bias to `T_meas`:

```
T_meas_aligned = T_meas · Exp([v_body; ω_body] · dt_skew_s)
```

This consumes `dt_skew_s` **with its sign** (positive shifts one way, negative
the other) instead of squaring it away. R-inflation is retained unchanged as the
residual-uncertainty term. Behind a flag, defaults conservative; falls back to
old behavior when `|dt_skew_s|` is below ~1 ms or skew confidence is low.

> This is a first-order approximation of full alignment — it corrects the EKF
> measurement only, not the Kabsch sample buffer. Phases 1–2 generalize it.

### Phase 1 — Reference pose history ring buffer

`CollectSample` currently discards everything but the latest paired read. To
interpolate the lighthouse stream backward we need recent reference history.

- Add `std::deque<TimedPose>` for the **reference** stream (timestamp + pose).
  ~150 ms depth is plenty (skew clamp is `[−40, +80] ms`).
- Push the raw reference pose+timestamp into the ring at the top of
  `CollectSample`, before the `Sample` is constructed.
- Cap the deque by age, drop the front past the window.

### Phase 2 — De-skew at sample construction (single chokepoint)

In `CollectSample`, build the `Sample` from a **time-aligned** reference:

```
t_target_phys = glfwGetTime() − dt_skew_s        // SLAM pose belongs to the past
ref_aligned   = InterpolateRef(t_target_phys)    // SLERP rot + LERP trans
Sample(ref_aligned, ConvertPose(target), t_target_phys)
```

- `InterpolateRef`: `std::lower_bound` on the ring, decoupled SLERP+LERP between
  the two bracketing entries.
- Gap gate (Phase research #4): bracket span > ~3× tick ⇒ skip de-skew (use raw
  simultaneous read) or mark sample invalid.
- `|dt_skew_s| < ~1 ms` or low skew confidence ⇒ no interpolation (never *add*
  interpolation noise for zero gain).
- Sign: `skew > 0` interpolates ref **backward** (history lookup, the common
  case). `skew < 0` (SLAM leads) needs ref slightly in the future ⇒ clamp to
  newest ref sample or extrapolate one tick (rare; ALVR overprediction).

Once Phase 2 is in, every consumer is fed simultaneous poses and the Phase-0
EKF-only correction is subsumed (remove or keep as belt-and-suspenders).

### Phase 3 — Reconcile R-inflation

With true alignment the systematic `v·skew` mismatch is largely gone, so
`k_skew` should down-weight only the **residual** (interpolation error +
`δskew`). Decide between:

- drive R-inflation off `δskew` (skew-estimate uncertainty) instead of full
  `skew`, or
- lower `k_skew` / `k_skew_rot` gains and let alignment carry the load.

Needs measurement (Phase 4), not a guess.

### Phase 4 — Validation

- A/B RMS calibration error with de-skew ON vs OFF during a scripted head-shake
  at known speed, swept over injected skews.
- Post-alignment correlation residual and confidence should drop if working.
- Watch the dropped-frame interpolation failure mode (reuse gap rejection).
- Log both `skew` and post-alignment residual for regression tracking.

---

## 7. Open questions to resolve before Phase 1

1. **Per-pose driver timestamps.** `DriverPose_t` carries `poseTimeOffset`. If the
   SLAM driver reports a real per-pose capture time, that is *better than a single
   global estimated skew* and could partially supersede the interpolation scheme.
   Verify availability before building on the single-skew assumption.

2. **Which device's velocity** drives the de-skew — the HMD's, or the target
   puck's? The skew is between *streams*, but the motion that contaminates a given
   pair is the rig's world motion over `skew`. Using HMD `vecVelocity` is the
   pragmatic proxy; confirm it tracks the target device closely enough when the
   target is a controller, not the HMD.

3. **Scale/de-skew ordering** (research #6): confirm per-axis scale is applied to
   `v` before the `v·dt` correction, or quantify the second-order residual if not.

---

## 8. File map

| File | Role |
| --- | --- |
| `src/overlay/Calibration.cpp` | `CollectSample` (capture point), `EstimateTimeSkew`, velocity FD, `SlamFixDriftStep` call site |
| `src/overlay/CalibrationCalc.cpp` | Kabsch, translation, per-axis scale, `SlamFixDriftStep` (`T_meas`), recenter/refine |
| `src/overlay/CalibrationCalc.h` | `Sample`/`Pose` structs, public API |
| `src/overlay/DriftFilter.cpp` | EKF predict/update, R-inflation (`dt_skew_s` consumer) |
| `src/overlay/DriftFilter.h` | `params.dt_skew_s`, `k_skew*` gains |
