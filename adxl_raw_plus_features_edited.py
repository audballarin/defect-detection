import asyncio
import websockets
import json
import csv
import time
import threading
import requests
from datetime import datetime
import numpy as np
from scipy.stats import kurtosis

# API / websocket connection
ip_addr = "192.168.50.10"    # Raspberry Pi running Klipper & Moonraker
moonraker_port = 7125
sensor = "adxl345"
moonraker = f"http://{ip_addr}:{moonraker_port}"
websocket = f"ws://{ip_addr}:{moonraker_port}/klippysocket"


window = 1000
step = 200

model_name = "snowflake"  
run_number = 12          # edit number for each new run
print_id = f"{model_name}_{run_number}"  

# edit print_state and state_flag before each print run
print_state = "u"       # 'u' for unclogged, 'c' for clogged, etc
state_flag = 1  # state_flag: 1 = extruding, 0 = full clog / no extrusion

name = print_id
raw_output_file = name + "_raw.csv"
feat_output_file = name + "_features.csv"
status_output_file = name + "_status.csv"

# tracker -- shared across threads so accelerometer logging & feature computation is synced to print start/end time
class StartTracker:
    def __init__(self):
        self.printing_started = False
        self.start_time = None
        self.stop_logging = False   # set to True to stop adxl_logger
        self.lock = threading.Lock()

shared = StartTracker()

# Polling Moonraker REST API & Logging Print Status Data (TIMESTAMPS, CURRENT/TARGET TEMPS, PROGRESS)
def log_status(shared):
    """
    After a print job has been launched, the API doesn't distinguish whether the printer is actively printing or just warming up/preparing.
    So I added my own printing_flag that bases the actual printing status on whether the target extruder temperature (eg 220 C) has been met
    (if the current temp is <220 then it's still warming up), and whether the target extruder temperature has gone down to 0 (ie finished printing).
    """
    status_file = open(status_output_file, "w", newline="")
    writer = csv.writer(status_file)
    writer.writerow(["timestamp", "bed_temp", "bed_target", "extruder_temp", "extruder_target", "progress", "printing_flag"])
    has_triggered = False
    print("Starting print status logging")

    while True:
        try:
            r = requests.get(f"{moonraker}/printer/objects/query?heater_bed&extruder&print_stats", timeout=5)
            data = r.json()["result"]["status"]

            bed_current = data["heater_bed"]["temperature"]
            bed_target = data["heater_bed"]["target"]
            nozzle_current = data["extruder"]["temperature"]
            nozzle_target = data["extruder"]["target"]
            progress = data["print_stats"].get("progress", 0)

            # detect when temps reach targets (which means printing has started)
            if (not has_triggered and
                bed_target > 0 and nozzle_target > 0 and
                bed_current >= bed_target and nozzle_current >= nozzle_target):
                with shared.lock:
                    shared.printing_started = True
                    shared.start_time = datetime.now()
                has_triggered = True
                print(f"[{shared.start_time}] Printing started (temps reached targets)")

            # if we've started printing and extruder target goes down to 0, print is finished
            if has_triggered and nozzle_target == 0:
                with shared.lock:
                    shared.stop_logging = True
                writer.writerow([datetime.now().isoformat(), bed_current, bed_target, nozzle_current, nozzle_target, progress, shared.printing_started])
                status_file.flush()
                print("Detected extruder target=0 after printing started ; stop accelerometer logging.")
                status_file.close()
                return

            writer.writerow([datetime.now().isoformat(), bed_current, bed_target, nozzle_current, nozzle_target, progress, shared.printing_started])
            status_file.flush()
            time.sleep(2)

        except Exception as e:
            print(f"Error accessing Moonraker REST API: {e}")
            time.sleep(5)

# ROLLING FEATURE COMPUTATION
def compute_features(x, y, z):
    """
    Given a window with multiple rows of values for each of x, y, z axis, compute the statistical features within that window.
    """
    features = {}

    # convert to np arrays for later functions
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # per-axis stats
    for axis_label, arr in zip(['x', 'y', 'z'], [x, y, z]):
        if len(arr) < 10:
            features.update({
                f"{axis_label}_rms": np.nan,
                f"{axis_label}_std": np.nan,
                f"{axis_label}_mean": np.nan,
                f"{axis_label}_kurt": np.nan,
                f"{axis_label}_crest": np.nan
            })
            continue

        rms = np.sqrt(np.mean(arr ** 2))
        std = np.std(arr)
        mean = np.mean(arr)
        krt = kurtosis(arr, fisher=False) 
        peak = np.max(np.abs(arr))
        crest = peak / rms if rms > 0 else np.nan

        features.update({
            f"{axis_label}_rms": float(rms),
            f"{axis_label}_std": float(std),
            f"{axis_label}_mean": float(mean),
            f"{axis_label}_kurt": float(krt),
            f"{axis_label}_crest": float(crest)
        })

    # magnitude per axis (mean of abs())
    features["x_magn"] = float(np.mean(np.abs(x))) if len(x) > 0 else np.nan
    features["y_magn"] = float(np.mean(np.abs(y))) if len(y) > 0 else np.nan
    features["z_magn"] = float(np.mean(np.abs(z))) if len(z) > 0 else np.nan

    # vector euclidean norm for combined magnitude across axes
    mag = np.sqrt(x**2 + y**2 + z**2)
    features["c_magn"] = float(np.mean(mag)) if len(mag) > 0 else np.nan

    # compute the rest of the combined-axis stats
    if len(x) >= 10:
        # euclidean norm for most stats (except kurtosis & crest factor)
        c_rms = np.sqrt(features["x_rms"]**2 + features["y_rms"]**2 + features["z_rms"]**2)
        c_std = np.sqrt(features["x_std"]**2 + features["y_std"]**2 + features["z_std"]**2)
        c_mean = np.sqrt(features["x_mean"]**2 + features["y_mean"]**2 + features["z_mean"]**2)

        c_kurt = float(kurtosis(mag, fisher=False))
        peak_mag = np.max(mag) if len(mag) > 0 else np.nan
        c_crest = float(peak_mag / c_rms) if (c_rms > 0 and not np.isnan(peak_mag)) else np.nan

        features.update({
            "c_rms": float(c_rms),
            "c_std": float(c_std),
            "c_mean": float(c_mean),
            "c_kurt": c_kurt,
            "c_crest": c_crest
        })
    else:
        features.update({
            "c_rms": np.nan,
            "c_std": np.nan,
            "c_mean": np.nan,
            "c_kurt": np.nan,
            "c_crest": np.nan
        })

    return features

# ADXL LOGGER
async def adxl_logger(shared):
    await asyncio.sleep(1)
    try:
        async with websockets.connect(websocket) as ws:
            print("Connected to websocket")
            # Request continuous adxl feed
            await ws.send(json.dumps({
                "id": 1,
                "method": "adxl345/dump_adxl345",
                "params": {"sensor": sensor}
            }))

            # Open files
            raw_file = open(raw_output_file, "w", newline="")
            raw_writer = csv.writer(raw_file)
            raw_writer.writerow(["timestamp", "x", "y", "z"])

            feat_file = open(feat_output_file, "w", newline="")
            feat_writer = csv.writer(feat_file)
            # Header: timestamp, print metadata, per-axis stats, magnitudes, combined stats
            feat_writer.writerow([
                "timestamp",
                "print_id",
                "print_state",
                "state_flag",
                "x_rms", "y_rms", "z_rms",
                "x_std", "y_std", "z_std",
                "x_mean", "y_mean", "z_mean",
                "x_kurt", "y_kurt", "z_kurt",
                "x_crest", "y_crest", "z_crest",
                "x_magn", "y_magn", "z_magn", "c_magn",
                "c_rms", "c_std", "c_mean", "c_kurt", "c_crest"
            ])

            buffer_x, buffer_y, buffer_z = [], [], []
            collecting = False
            sample_counter = 0

            while True:
                # Check stop flag before attempting to receive more samples
                with shared.lock:
                    if shared.stop_logging:
                        print("Stop flag detected before reading from websocket -> closing files and exiting adxl_logger.")
                        raw_file.close()
                        feat_file.close()
                        return

                # Receive a websocket message (may contain multiple samples)
                try:
                    msg = await ws.recv()
                except websockets.ConnectionClosed:
                    print("Websocket connection closed by server.")
                    break

                try:
                    data = json.loads(msg)
                except Exception as e:
                    print(f"Malformed websocket message (not JSON): {e}")
                    continue

                if "params" in data and "data" in data["params"]:
                    for row in data["params"]["data"]:
                        # row should be [timestamp, x, y, z]
                        try:
                            t, x, y, z = row
                        except Exception:
                            # Unexpected row format: skip
                            continue

                        # Check if printing has started; if not, skip samples
                        with shared.lock:
                            start = shared.printing_started
                            stop_now = shared.stop_logging
                        if not start:
                            continue
                        if stop_now:
                            print("Stop flag detected inside processing loop -> closing files and returning.")
                            raw_file.close()
                            feat_file.close()
                            return

                        # Start collecting now that printing has started
                        if not collecting:
                            print(f"ADXL logging started at {datetime.now().isoformat()}")
                            collecting = True

                        # Write raw sample
                        raw_writer.writerow([t, x, y, z])
                        raw_file.flush()
                        sample_counter += 1

                        # Update buffers and maintain window size
                        buffer_x.append(x)
                        buffer_y.append(y)
                        buffer_z.append(z)
                        if len(buffer_x) > window:
                            buffer_x = buffer_x[-window:]
                            buffer_y = buffer_y[-window:]
                            buffer_z = buffer_z[-window:]

                        # Compute and write features every `step` samples (and require at least half-window)
                        if (sample_counter % step == 0) and (len(buffer_x) >= window // 2):
                            feats = compute_features(np.array(buffer_x), np.array(buffer_y), np.array(buffer_z))

                            # Build row in exact header order
                            row = [
                                datetime.now().isoformat(),  # timestamp
                                print_id,
                                print_state,
                                state_flag,
                                # per-axis rms
                                feats.get("x_rms", np.nan), feats.get("y_rms", np.nan), feats.get("z_rms", np.nan),
                                # per-axis std
                                feats.get("x_std", np.nan), feats.get("y_std", np.nan), feats.get("z_std", np.nan),
                                # per-axis mean
                                feats.get("x_mean", np.nan), feats.get("y_mean", np.nan), feats.get("z_mean", np.nan),
                                # per-axis kurtosis
                                feats.get("x_kurt", np.nan), feats.get("y_kurt", np.nan), feats.get("z_kurt", np.nan),
                                # per-axis crest
                                feats.get("x_crest", np.nan), feats.get("y_crest", np.nan), feats.get("z_crest", np.nan),
                                # magnitudes
                                feats.get("x_magn", np.nan), feats.get("y_magn", np.nan), feats.get("z_magn", np.nan), feats.get("c_magn", np.nan),
                                # combined stats
                                feats.get("c_rms", np.nan), feats.get("c_std", np.nan), feats.get("c_mean", np.nan), feats.get("c_kurt", np.nan), feats.get("c_crest", np.nan)
                            ]

                            feat_writer.writerow(row)
                            feat_file.flush()

    except Exception as e:
        print(f"Exception in adxl_logger: {e}")
    finally:
        # Try to close files if still open
        try:
            raw_file.close()
        except Exception:
            pass
        try:
            feat_file.close()
        except Exception:
            pass
        print("ADXL logger exiting.")


def main():
    # Start status thread first (so it can detect the printing start)
    t1 = threading.Thread(target=log_status, args=(shared,), daemon=True)
    t1.start()

    # Run the ADXL logger (async)
    try:
        asyncio.run(adxl_logger(shared))
    except KeyboardInterrupt:
        # Allow Ctrl-C to cleanly stop
        with shared.lock:
            shared.stop_logging = True
        print("KeyboardInterrupt received. Signaled stop_logging and exiting.")
    except Exception as e:
        print(f"Unhandled exception in main: {e}")

if __name__ == "__main__":
    main()
