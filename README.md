# Store Customer Tracker

> Real-time retail analytics using computer vision, OCR, and re-identification  

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![YOLOv8](https://img.shields.io/badge/detection-YOLO11-orange) ![License](https://img.shields.io/badge/license-MIT-green)

---

<h2>Description</h2>
This project processes surveillance video from a Pearle Vision and uses it to analyze trends in customer behavior using computer vision. It tracks customers in real-time, determining both the total time they spend in the store and the time they spend browsing specific glasses. It also logs entries/exits with timestamps, finally putting all the information into a CSV file. I also save the results to graphs in order to view the data very easily. To ensure accuracy, I incorporated person re-identification so that the results are much more accurate.
<br><br>
The goal is to analyze customer behavior into insights, including things like visit duration and zone engagement, for use in business analytics and increasing store traffic.
<br />

<h2>Features </h2>

- <b>Person Detection:</b> Uses YOLO11 for real-time person detection
- <b>Person Re-Identification:</b> Uses PyTorch and TorchREID to do person re-id
- <b>Zone-Based Logic:</b>
  - Entry/exit detection with customizable polygon zones
  - Glasses browsing zone tracking
  - Worker zone filtering to ignore irrelevant detections
- <b>OCR Timestamps:</b> Pulls timestamps from the video feed using Tesseract OCR
- <b>CSV Logging:</b> Outputs detailed customer logs with entry time, exit time, total duration, and more
- <b>MatPlotLib Graphs:</b> Creates graphs from the data collected


<h2>Tech Stack</h2>

- <b>Python</b>
- <b>PyTorch</b> 
- <b>OpenCV</b>
- <b>Ultralytics YOLO11</b>
- <b>Supervision</b> (Roboflow's tracking & annotation toolkit)
- <b>Numpy</b>
- <b>Pytesseract</b> (Tesseract OCR)
- <b>MatPlotLib</b>

<h2>How it works:</h2>

<p align="center">
Add the Security Footage Video/Place to output video: <br/>
<img src="https://i.imgur.com/hHsbMHk.png" height="80%" width="80%" alt="add footage"/>
<br />
<br />
View the csv report:  <br/>
<img src="https://i.imgur.com/Urzh0TS.png" height="80%" width="80%" alt="CSV report"/>
<br />
<br />
View the generated Graphs:  <br/>
<img src="https://i.imgur.com/jPdFh6u.png" height="80%" width="80%" alt="Peak customer hours"/>
<br />
<br />
View the generated Graphs:  <br/>
<img src="https://i.imgur.com/8GL7z6P.png" height="80%" width="80%" alt="Customer visit duration"/>
<br />


<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
