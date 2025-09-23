# defect-detection
# Vibration-Data-Driven Detection of 3D Printing Defects

Description of the project.
The process of 3D printing produces distinct vibrational patterns. This project would investigate vibration data from 3D printing processes using an accelerometer mounted on the printer. The goal is to train a model that can identify patterns in vibration signals indicating print defects, such as a clogged nozzle or underextrusion. 

Clear goal(s) (e.g. Successfully predict the number of students attending lecture based on the weather report).
The main goal is to train a model that can distinguish between normal prints and defective prints based on vibration data from the 3D printer. 

What data needs to be collected and how you will collect it (e.g. scraping xyz website or polling students).
The data will be collected by mounting a low-cost accelerometer (ADXL345) to the print head of a 3D printer (Ender 5 Plus) and recording vibration signals from different types of prints. The dataset will include normal prints and prints with defects that are intentionally induced by modifying certain print settings (e.g. inducing over/underextrusion by increasing or decreasing the flow rate setting, or inducing a clog by significantly reducing the temperature). Each print will be labeled according to the outcome. 

How you plan on modeling the data (e.g. clustering, fitting a linear model, decision trees, XGBoost, some sort of deep learning method, etc.).
There's a good amount of prior work in this area using a variety of modeling techniques on vibration data to detect different types of defects. I would begin by extracting certain statistical features commonly used in vibration analysis (such as  mean, standard deviation, root mean square, kurtosis index, etc.). An example of existing work in this area uses a support vector machine variant to detect normal versus clogged printer states [Li et al. 2019], so that may be my first approach. (To clarify, the authors' code is not open-source or available so I would be implementing my own version of this technique.)

How do you plan on visualizing the data? (e.g. interactive t-SNE plot, scatter plot of feature x vs. feature y).
The final visualizations would depend on the modeling approach taken, but some of the exploratory visualizations would include spectrograms showing the frequency ranges of the data and plots comparing specific statistical features (e.g. RMS) for different prints in the dataset. 

What is your test plan? (e.g. withhold 20% of data for testing, train on data collected in October and test on data collected in November, etc.). 
I will spend October collecting the dataset for training, which will include 3-4 different 3D models/geometries for generalizability, and a subtotal of about 50 (small / <20min) training prints. By midterm I will have preliminary results (e.g. extracted features and basic fitting). In November I will test on ~12 more prints (20% of total data collected), and will include one new 3D model among these to test generalizability to geometries not included in training. 


Reference:
Li, Yongxiang, et al. "In-situ monitoring and diagnosing for fused filament fabrication process based on vibration sensors." Sensors 19.11 (2019): 2589.

