import pandas as pd
import numpy as np
from scipy.stats import kurtosis

WINDOW = 1000   # window consists of 1000 samples, which is ~1/3 of a second
STEP = 200  # step size for rolling windows is 200 samples, so there's ~80% overlap between consecutive samples

# Given a window with multiple rows of values for each of x, y, z axis,
#   compute the statistical features within that window

def compute_window_features(x, y, z):
    features = {}

    # turn the collected values for each axis to np array
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Per-axis statistics
    for axis, arr in zip(["x", "y", "z"], [x, y, z]):
        if len(arr) < 10:
            features[f"{axis}_rms"] = np.nan
            features[f"{axis}_std"] = np.nan
            features[f"{axis}_mean"] = np.nan
            features[f"{axis}_kurt"] = np.nan
            features[f"{axis}_crest"] = np.nan
            continue

        # define the statistical features that will be computed per axis
        rms = np.sqrt(np.mean(arr**2))
        std = np.std(arr)
        mean = np.mean(arr)
        krt = kurtosis(arr, fisher=False)
        peak = np.max(np.abs(arr))
        crest = peak / rms if rms > 0 else np.nan

        features[f"{axis}_rms"] = float(rms)
        features[f"{axis}_std"] = float(std)
        features[f"{axis}_mean"] = float(mean)
        features[f"{axis}_kurt"] = float(krt)
        features[f"{axis}_crest"] = float(crest)

    # magnitude for each axis is just the mean of abs()
    features["x_magn"] = float(np.mean(np.abs(x))) if len(x) else np.nan
    features["y_magn"] = float(np.mean(np.abs(y))) if len(y) else np.nan
    features["z_magn"] = float(np.mean(np.abs(z))) if len(z) else np.nan

    # vector euclidean norm for combined magnitude across axes
    mag = np.sqrt(x**2 + y**2 + z**2)
    features["c_magn"] = float(np.mean(mag)) if len(mag) else np.nan

    # compute the rest of the combined-axis stats
    if len(x) >= 10:
        # euclidean norm for most stats (except kurtosis & crest factor)
        features["c_rms"]  = float(np.sqrt(features["x_rms"]**2  + features["y_rms"]**2  + features["z_rms"]**2))
        features["c_std"]  = float(np.sqrt(features["x_std"]**2  + features["y_std"]**2  + features["z_std"]**2))
        features["c_mean"] = float(np.sqrt(features["x_mean"]**2 + features["y_mean"]**2 + features["z_mean"]**2))

        features["c_kurt"] = float(kurtosis(mag, fisher=False))

        peak_mag = np.max(mag)
        features["c_crest"] = float(peak_mag / features["c_rms"]) if features["c_rms"] > 0 else np.nan
    else:
        features["c_rms"] = np.nan
        features["c_std"] = np.nan
        features["c_mean"] = np.nan
        features["c_kurt"] = np.nan
        features["c_crest"] = np.nan

    return features


def compute_offline_features(
        raw_path,
        output_path,
        print_id,
        print_state="c",
        state_flag=0,
        window=WINDOW,
        step=STEP,
        status_path=None,
    ):

    raw = pd.read_csv(raw_path)

    # handling timestamps
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).copy()
    raw = raw.sort_values("timestamp")

    '''
    For some older raw accelerometer logs, I hadn't scripted an automatic cutoff to stop logging after printing ends,
    so there's some extra noisy data at the end from after the print had stopped. However, I had collected print status data 
    for those prints (i.e. timestamps, target/current temp, progress data).
    This part of the code optionally (if status_file is included as an arg) detects when to cut off the noisy extra accel logs at the end of the file.
    '''
    if status_path is not None:
        status = pd.read_csv(status_path, parse_dates=["timestamp"])
        cutoff_rows = status[status["extruder_target"] == 0]

        if len(cutoff_rows) == 0:
            print("WARNING: No extruder_target == 0 found. Using full file.")
            cutoff_ts = None
        else:
            cutoff_ts = cutoff_rows["timestamp"].min()
            print(f"Cutoff timestamp (print end): {cutoff_ts}")
            raw = raw[raw["timestamp"] <= cutoff_ts]

    # Calculate rolling windows
    rows = []
    buf_x, buf_y, buf_z = [], [], []

    for i, row in raw.iterrows():

        buf_x.append(row["x"])
        buf_y.append(row["y"])
        buf_z.append(row["z"])

        # Maintain fixed-size sliding window
        if len(buf_x) > window:
            buf_x = buf_x[-window:]
            buf_y = buf_y[-window:]
            buf_z = buf_z[-window:]

        if i % step == 0 and len(buf_x) >= window // 2:
            feats = compute_window_features(buf_x, buf_y, buf_z)
            out_row = {
                "timestamp": row["timestamp"],
                "print_id": print_id,
                "print_state": print_state,
                "state_flag": state_flag,
            }
            out_row.update(feats)
            rows.append(out_row)

    df = pd.DataFrame(rows)

    # preserve this specific column order (same as in the online logging file)
    final_cols = [
        "timestamp",
        "print_id", "print_state", "state_flag",
        "x_rms", "y_rms", "z_rms",
        "x_std", "y_std", "z_std",
        "x_mean", "y_mean", "z_mean",
        "x_kurt", "y_kurt", "z_kurt",
        "x_crest", "y_crest", "z_crest",
        "x_magn", "y_magn", "z_magn",
        "c_magn", "c_rms", "c_std", "c_mean", "c_kurt", "c_crest"
    ]
    df = df[final_cols]
    df.to_csv(output_path, index=False)
    print(f"Offline features saved to {output_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline feature calculator.")

    parser.add_argument("raw_path", help="Raw accelerometer CSV")
    parser.add_argument("output_path", help="Path to write features CSV")
    parser.add_argument("--print-id", help="Print ID", default="unknown")
    parser.add_argument("--print-state", default="c")
    parser.add_argument("--state-flag", type=int, default=0)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--status", help="Optional status CSV", default=None)


    args = parser.parse_args()

    compute_offline_features(
        raw_path=args.raw_path,
        output_path=args.output_path,
        print_id=args.print_id,
        print_state=args.print_state,
        state_flag=args.state_flag,
        window=args.window,
        step=args.step,
        status_path=args.status
    )

# Used this script previously to calculate features on a couple raw accelerometer dumps
#   that I hadn't done live/real-time feature calc on
'''
if __name__ == "__main__":
    compute_offline_features(
        raw_path="adxl_cube_clog0.csv",
        status_path="status_cube_clog0.csv",
        output_path="cube_clog0_all_features.csv",
        print_id="cube_12",
        print_state="c",
        state_flag=0
    )
    compute_offline_features(
        raw_path="adxl_frog_clog0.csv",
        status_path="status_frog_clog0.csv",
        output_path="frog_clog0_all_features.csv",
        print_id="frog_10",
        print_state="c",
        state_flag=0
    )
'''