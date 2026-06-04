"""
Validate a principled drift-per-meter estimator against real calibration logs.

Method: structure function (Allan-variance style).
  The raw cross-system offset offset(s) drifts as a random walk vs walked
  distance s. For a Wiener process observed under white noise R:
      D(L) = E[(offset(s+L) - offset(s))^2] = 2R + sigma_lin_pos_sq * L
  Fit a line over L -> slope = sigma_lin_pos_sq (the filter's drift rate),
  intercept = 2R. drift_per_meter[cm/m] = sqrt(slope) with offset in cm, L in m.

No floors, no NIS, no bias. Just measure the diffusion coefficient.
"""
import sys, glob, os
import numpy as np

LOG_DIR = os.path.expandvars(r"%USERPROFILE%\AppData\LocalLow\SpaceCalibrator\Logs")

def load(path):
    rows = []
    header = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = line.split(",")
                continue
            rows.append(line.split(","))
    idx = {name: i for i, name in enumerate(header)}
    def col(name):
        i = idx[name]
        out = np.empty(len(rows))
        for k, r in enumerate(rows):
            try:
                out[k] = float(r[i]) if i < len(r) and r[i] != "" else np.nan
            except ValueError:
                out[k] = np.nan
        return out
    t = col("Timestamp")
    raw = np.stack([col("posOffset_rawComputed.x"),
                    col("posOffset_rawComputed.y"),
                    col("posOffset_rawComputed.z")], axis=1)   # cm
    rel = np.stack([col("posOffset_byRelPose.x"),
                    col("posOffset_byRelPose.y"),
                    col("posOffset_byRelPose.z")], axis=1)      # cm
    v_lin = col("slamfix_v_lin_mm_s") / 1000.0                  # m/s
    return t, raw, rel, v_lin

def structure_fit(t, pos, v_lin, label):
    # cumulative walked distance (m), integrated from logged linear speed
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0, 0.1)
    dist = np.cumsum(np.maximum(v_lin, 0) * dt)
    # keep finite rows
    good = np.isfinite(pos).all(axis=1) & np.isfinite(dist) & np.isfinite(v_lin)
    pos, dist, v_lin = pos[good], dist[good], v_lin[good]
    total = dist[-1] - dist[0] if len(dist) else 0.0
    if total < 1.0:
        print("  [%s] only %.2f m walked - skip" % (label, total))
        return
    # resample offset onto a uniform distance grid so structure-function lags
    # are true distance lags (not time lags).
    ds = 0.02  # 2 cm grid
    grid = np.arange(dist[0], dist[-1], ds)
    pg = np.stack([np.interp(grid, dist, pos[:, k]) for k in range(3)], axis=1)
    # structure function over a range of distance lags
    lags_m = np.arange(0.10, 1.60, 0.05)
    D = []
    for L in lags_m:
        k = int(round(L / ds))
        if k < 1 or k >= len(pg):
            D.append(np.nan); continue
        diff = pg[k:] - pg[:-k]
        D.append(np.nanmean(np.sum(diff * diff, axis=1)))   # cm^2
    D = np.array(D)
    m = np.isfinite(D)
    # robust line fit D = 2R + slope*L over the linear region
    A = np.stack([np.ones(m.sum()), lags_m[m]], axis=1)
    coef, *_ = np.linalg.lstsq(A, D[m], rcond=None)
    intercept, slope = coef
    drift_cm_per_m = np.sqrt(max(slope, 0.0))   # cm/m
    R_static_cm2 = max(intercept, 0.0) / 2.0
    print("  [%s] walked=%.1fm  slope=%.2f cm^2/m  =>  drift=%.2f cm/m   (R_static=%.2f cm, intercept=%.2f)"
          % (label, total, slope, drift_cm_per_m, np.sqrt(R_static_cm2), intercept))
    return drift_cm_per_m

def main():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "spacecal_log.*.txt")))
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    for path in files:
        print(os.path.basename(path))
        t, raw, rel, v_lin = load(path)
        print("  samples=%d  speed>0.1m/s frac=%.2f" %
              (len(t), np.mean(v_lin > 0.1)))
        structure_fit(t, raw, v_lin, "rawComputed")
        structure_fit(t, rel, v_lin, "byRelPose")

if __name__ == "__main__":
    main()
