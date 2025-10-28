import asyncio
import websockets
import json
import csv
import time
import threading
import queue
import requests
from datetime import datetime
import numpy as np
from scipy.stats import kurtosis

# API CONFIGURATION
ip_addr = "192.168.50.10"    # assigned IP address of the Raspberry Pi running the Klipper firmware (and connected to the accelerometer)
moonraker_port = 7125      # Moonraker is the web server that exposes the Klipper API & gives access to accelerometer output  
sensor = "adxl345"      # this is the name of the sensor that I added to the Klipper configuration file on my Raspberry Pi
moonraker = f"http://{ip_addr}:{moonraker_port}"       # to access regular Moonrake REST API for basic print status info
websocket = f"ws://{ip_addr}:{moonraker_port}/klippysocket"    # websocket for accessing API endpoints that are not available through http/REST-type API

# OUTPUT FILES
name = "snowflake_partial1"     # update name/number with each print/3D model
#date = "17oct"
raw_output_file = name + "_raw.csv"
feat_output_file = name + "_features.csv"
status_output_file = name + "_status.csv"

# window sizes for computing rolling statistical features
window = 1000      
step = 200       

'''
In this code I use two parallel threads for logging 
    (1) accelerometer data from the Moonraker websocket,
    (2) regular print status data (e.g. current/target temperature, progress, etc) from Moonraker's regular http/REST API
The purpose of threading is that I want to start logging accelerometer data at the exact time that the printer actually 
starts printing/extruding, not just preparing. The status endpoint says "printing" even when the printer is just warming up, 
so instead I check when the printer meets its target temperatures, and use that as the global start time to trigger the accelerometer log. 
'''
class StartTracker:
    def __init__(self):
        self.printing_started = False
        self.start_time = None
        self.lock = threading.Lock()
shared = StartTracker()


def log_status(shared):
    # Log params & check current vs target temperature from REST API to flag when printing actually starts
    status_file = open(status_output_file, "w", newline="")
    writer = csv.writer(status_file)
    writer.writerow(["timestamp", "bed_temp", "bed_target", "extruder_temp", "extruder_target", "progress", "printing_flag"])
    has_triggered = False
    print("Starting print status logging")

    while True:
        try:
            r = requests.get(f"{moonraker}/printer/objects/query?heater_bed&extruder&print_stats")
            data = r.json()["result"]["status"]

            bed_current = data["heater_bed"]["temperature"]
            bed_target = data["heater_bed"]["target"]
            nozzle_current = data["extruder"]["temperature"]
            nozzle_target = data["extruder"]["target"]
            progress = data["print_stats"].get("progress", 0)

            # Detect when temps reach targets
            if not has_triggered and bed_current >= bed_target and nozzle_current >= nozzle_target and bed_target > 0 and nozzle_target > 0:
                with shared.lock:
                    shared.printing_started = True
                    shared.start_time = datetime.now()
                has_triggered = True
                print(f"[{shared.start_time}] Printing flag set TRUE (temps reached targets)")

            writer.writerow([datetime.now().isoformat(), bed_current, bed_target, nozzle_current, nozzle_target, progress, shared.printing_started])
            status_file.flush()
            time.sleep(2)

        except Exception as e:
            print(f"Error accessing Moonraker REST API: {e}")
            time.sleep(5)


def compute_features(x, y, z):
    features = {}
    for axis, arr in zip(['x', 'y', 'z'], [x, y, z]):
        if len(arr) < 10:
            # Not enough samples for stable stats
            features.update({f"{axis}_rms": np.nan, f"{axis}_std": np.nan,
                          f"{axis}_mean": np.nan, f"{axis}_kurt": np.nan,
                          f"{axis}_crest": np.nan})
            continue

        rms = np.sqrt(np.mean(arr**2))
        std = np.std(arr)
        mean = np.mean(arr)
        krt = kurtosis(arr, fisher=False) 
        crest = np.max(np.abs(arr)) / rms if rms > 0 else np.nan

        features.update({
            f"{axis}_rms": rms,
            f"{axis}_std": std,
            f"{axis}_mean": mean,
            f"{axis}_kurt": krt,
            f"{axis}_crest": crest
        })

    return features


async def adxl_logger(shared):
    # Connect to Moonraker websocket, log ADXL data + rolling features
    await asyncio.sleep(1)
    async with websockets.connect(websocket) as ws:
        print("Connected to websocket")
        # polling the websocket for the dump_adxl345 endpoint requires sending a json message in the following format:
        await ws.send(json.dumps({
            "id": 1,
            "method": "adxl345/dump_adxl345",
            "params": {"sensor": "adxl345"}
        }))

        # CSV for raw accelerometer data sampled at 3200Hz
        raw_file = open(raw_output_file, "w", newline="")
        raw_writer = csv.writer(raw_file)
        raw_writer.writerow(["timestamp", "x", "y", "z"])

        # CSV for rolling statistical features (RMS, standard dev, etc); lower output data rate, hence the separate file from the accelerometer log
        feat_file = open(feat_output_file, "w", newline="")
        feat_writer = csv.writer(feat_file)
        feat_writer.writerow([
            "timestamp",
            "x_rms", "y_rms", "z_rms",
            "x_std", "y_std", "z_std",
            "x_mean", "y_mean", "z_mean",
            "x_kurt", "y_kurt", "z_kurt",
            "x_crest", "y_crest", "z_crest"
        ])

        buffer_x, buffer_y, buffer_z = [], [], []
        collecting = False
        sample_counter = 0

        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if "params" in data and "data" in data["params"]:
                for row in data["params"]["data"]:
                    t, x, y, z = row
                    with shared.lock:
                        start = shared.printing_started
                    if not start:
                        continue

                    if not collecting:
                        print(f"ADXL logging started at {datetime.now()}")
                        collecting = True

                    # Write raw
                    raw_writer.writerow([t, x, y, z])
                    sample_counter += 1

                    # Update buffers
                    buffer_x.append(x)
                    buffer_y.append(y)
                    buffer_z.append(z)

                    # Maintain window
                    if len(buffer_x) > window:
                        buffer_x = buffer_x[-window:]
                        buffer_y = buffer_y[-window:]
                        buffer_z = buffer_z[-window:]

                    # Compute features periodically
                    if sample_counter % step == 0 and len(buffer_x) >= window // 2:
                        feats = compute_features(np.array(buffer_x), np.array(buffer_y), np.array(buffer_z))
                        feat_writer.writerow([datetime.now().isoformat(), *feats.values()])
                        feat_file.flush()

                    raw_file.flush()


def main():
    t1 = threading.Thread(target=log_status, args=(shared,), daemon=True)
    t1.start()
    asyncio.run(adxl_logger(shared))

if __name__ == "__main__":
    main()
