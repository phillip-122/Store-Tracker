<h1> Store Customer Tracker</h1>

<h2>Description</h2>
This project processes surveillance video from a Pearle Vision and uses it to analyze trends in customer behavior using computer vision. It tracks customers in real-time, determining both the total time they spend in the store and the time they spend browsing specific glasses. It also logs entries/exits with timestamps, finally putting all the information into a CSV file.
<br><br>
The goal is to analyze customer behavior into insights, including things like visit duration and zone engagement, for use in business analytics and increasing store traffic.
<br />

<h2>Features </h2>

- <b>Person Detection:</b> Uses YOLO11 for real-time person detection
- <b>Zone-Based Logic:<b>
  - Entry/exit detection with customizable polygon zones
  - Glasses browsing zone tracking
  - Worker zone filtering to ignore irrelevant detections
- <b>OCR Timestamps:<b> Pulls timestamps from the video feed using Tesseract OCR
- <b>CSV Logging:<b> Outputs detailed customer logs with entry time, exit time, total duration, and more


<h2>Tech Stack</h2>

- <b>Python</b> 
- <b>OpenCV</b>
- <b>Ultralytics YOLO11</b>
- <b>Supervision</b> (Roboflow's tracking & annotation toolkit)
- <b>Numpy</b>
- <b>Pytesseract</b> (Tesseract OCR)

<h2><b>THIS IS STILL A WORK IN PROGRESS I WILL BE UPDATING THE INFO HERE BUT I AM STILL WORKING ON THIS</b></h2>
<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
