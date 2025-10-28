# defect-detection
# Midterm Report: Vibration Data-Driven Detection of 3D Printing Defects


## Background & Motivation Recap
- The process of 3D printing produces **distinct vibrational profiles** that are most prominent near the nozzle and on the print bed, and can be detected using an accelerometer
- These vibrational profiles can **change when defects or anomalies** occur 
- This project specifically aims to **collect and model accelerometer data** for detecting 3D prints with a **fully clogged nozzle** (where there is no extrusion at all) vs. **partially clogged nozzle** (partial, under-extrusion) vs. **normal nozzle condition**
- This is based on existing research on 3D print monitoring from accelerometer data, including:
  - Li, Yongxiang, et al. "In-situ monitoring and diagnosing for fused filament fabrication process based on vibration sensors." Sensors 19.11 (2019): 2589. 
  - Isiani A, Weiss L, Bardaweel H, Nguyen H, Crittenden K. Fault Detection in 3D Printing: A Study on Sensor Positioning and Vibrational Patterns. Sensors (Basel). 2023 Aug 30;23(17):7524.
  
## Approach & Experimental Setup for Dataset Collection
- The first step was setting up the accelerometer data collection system, which was a fairly involved process with several hiccups (mostly firmware-related) but is now running smoothly!
- Hardware/firmware: 
  - Ender 5 Plus with Klipper/Moonraker firmware running on a Raspberry Pi 3B+
  - ADXL345 accelerometer
- Mounted accelerometer using [this 3D printed mount](https://www.printables.com/model/343758-adxl345-bltouchcrtouch-mount/comments) from Printables screwed to printer (insert photo)
- Wired accelerometer to Raspberry Pi & enabled SPI connection on Raspberry Pi (originally tried standard I2C connection but it had too low throughput for capturing high frequency vibrations; I could only get up to ~120 Hz whereas SPI goes up to 3200 Hz)
Updated printer.cfg file in Klipper firmware (via Fluidd interface) 
- Enabled API socket based on [instructions](https://www.klipper3d.org/API_Server.html)
- While basic info such as print status can be accessed with HTTP REST API, accessing more advanced Moonraker endpoints, especially for large streamed data including the adxl345/adxl345_dump endpoint, requires websocket client — used websocat client in terminal to sanity-check the adxl345_dump endpoint stream, and then websockets package in Python script to log and write the data to CSV file
  - % websocat ws://192.168.50.10:7125/klippysocket
  - Then enter {"id": 1, "method": "adxl345/dump_adxl345", "params": {"sensor": "adxl345"}}
- Obtained API authorization key prior to this 
- Wrote accelerometer data logging script in Python 
- **Dataset Design** — In my original proposal, I was focused on getting many iterations of prints for a few geometries. However, since reading literature in this area that uses a relatively small number of prints to train models, as well as setting up a sensor polling connection with a very high output data rate, I've decided to focus less on maximizing the number of iterations of a given print in the dataset and more on maximizing **variation in geometries** in the dataset
  
## Data Logging, Processing, and Cleaning
- The adxl345/adxl345_dump endpoint gives raw accelerometer readings
  - The format of each entry is time, x_acceleration, y_acceleration, z_acceleration (e.g., 61895.221076, 74.020594, 296.082377, -9652.803275)
- The sampling rate of the accelerometer data is 3200 Hz, which corresponds to an output rate of about 4 million raw data points per 25 minutes of printing.
- My data logging script **outputs three different files**:
  - **1. Raw accelerometer values** (at maximum output data rate; thousands of data points per second)
  - **2. Rolling statistical features** computed over windows of every 1000 data points (with a step size of 200 data points)
    - This includes, on each of the X, Y, and Z axes: root mean square (RMS), mean, standard deviation, crest factor, and kurtosis
  - **3. Print status information** from the API, including progress, current/target temperatures
    - The print status log begins first, and there’s a **shared flag mechanism** in the script that starts the accelerometer data log when the script detects that the actual printing has started (i.e. not just preparing), which happens when current temperatures == target temperatures. (Originally used a global variable for this but then switched to threading implementation)
- After the logs are collected, I do **time synchronization**; for the raw sensor data, I normalize & convert the Raspberry Pi timestamps (that represent the time since the Pi was last booted) to more useful H:M:S timestamps starting at 00:00:00

## Preliminary Visualizations
- The most straightforward way to visualize dense accelerometer data is with a **spectrogram**, computed using Short-Time Fourier Transform, which breaks down raw signals into a range of different frequencies and visualizes the strength of different frequency bands
- As we can see in these example of spectrograms of the **magnitude** (calculated w/ **Euclidean distance** formula) of the acceleration signals for a fully clogged (zero extrusion) snowflake vs partially clogged (weak extrusion) snowflake, there are some differences but they're not particularly understandable — hence the need for feature engineering and modeling 


- We can also plot the statistical features, such as RMS, for individual prints. Similarly, the differences are not obvious without further modeling 

## Data Modeling & Preliminary Results
- The main modeling done so far is constructing the **feature vector matrix** with statistical features identified by Li, Yongxiang, et al. (2019) as useful for detecting defects:
  - For each of the X, Y, Z axes: root mean square (RMS), mean, standard deviation, kurtosis, and crest factor
- Then, to identify which of these features is the most useful for distinguishing states, I ran **Principal Component Analysis** 
- First, comparing individual fully vs. partially clogged 3d prints:


- Then, combining multiple prints into a larger dataset
 
  - We can see that while the features tend to be similar in fully clogged vs partially clogged prints, there is more variance along both PC1 and PC2 in a partially clogged print with extrusion. Of the features, X_rms is consistently the biggest contributor to PC1, which at around 0.39 is a moderate contribution. So far the dataset only contains fully clogged and partially clogged prints — this is promising information as we begin incorporating data from fully normal prints. 

## Next Steps
- **Expand dataset** — the hardware, firmware, and sensor polling setup was an initial challenge that took some time to refine, but now we’re well-positioned to add more prints to the training dataset, especially normal prints. 
- **Perform PCA** on expanded dataset to confirm features of interest
- Train a support vector machine (SVM) 
