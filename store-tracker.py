import csv
import re
from ultralytics import YOLO
import supervision as sv
import cv2 as cv
import numpy as np
import pytesseract
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# filePath = "C:\\Users\\PHILLIP\\Desktop\\Code Projects\\python\\Store-Tracker\\Videos - Copy\\merged_trimmed_20250523.mp4"
filePath = r"C:\Users\PHILLIP\Downloads\merged_trimmed_short - Made with Clipchamp.mp4"
targetPath = "C:\\Users\\PHILLIP\\Desktop\\Code Projects\\python\\Store-Tracker\\output.mp4"

lineTop = sv.Point(x=200, y=0)
lineBottom = sv.Point(x=200, y=900)

workerZoneOne = np.array([
    [0, 1079],
    [0, 850],
    [940, 950],
    [970, 1079]
])

glassesZone = np.array([
    [660, 0],
    [1919, 0],
    [1919, 1050], 
    [1300, 1919],
    [970, 830],
    [660, 740]
])

legitEntryZone = np.array([
    [220, 0],
    [1919, 0],
    [1919, 1079],
    [220, 1079]
])


#for just date
# dateZone = np.array([
#     [1380, 1000],
#     [1650, 1000],
#     [1650, 1070],
#     [1380, 1070]
# ])

dateX, dateY = 1380, 1000
dateW, dateH = 270, 70

#for just time
# timeZone = np.array([
#     [1660, 1000],
#     [1880, 1000],
#     [1880, 1070],
#     [1660, 1070]
# ])

timeX, timeY = 1660, 1000
timeW, timeH = 220, 70

def secondsToString(seconds):

    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    secondsString = (f"{hours} hours {minutes} minutes {seconds} seconds")

    return secondsString

def totalTimeCalc(entryTime, exitTime):
    totalDuration = {}
    totalDurationSeconds = 0

    for tracker_id in entryTime.keys() | exitTime.keys(): #It expects the entry/exit times dict to be in {id: HH:MM:SS} format
        entryTimeString = entryTime.get(tracker_id)
        exitTimeString = exitTime.get(tracker_id)
        formatString = "%H:%M:%S"
        totalSeconds = 0

        if entryTimeString and exitTimeString:
            try:
                entry = datetime.strptime(entryTimeString, formatString)
                exit = datetime.strptime(exitTimeString, formatString)
                duration = exit - entry

                #This is so I can format the output so it says _ Hours _ Minutes _ Seconds

                totalSeconds = int(duration.total_seconds())
                secondsString = secondsToString(totalSeconds)

                totalDuration[tracker_id] = secondsString

                totalDurationSeconds += totalSeconds
                
            except Exception as e:
                print(f"Error getting times for ID {tracker_id}: {e}")
        else:
            totalDuration[tracker_id] = "N/A"

    return totalDuration, totalDurationSeconds



if __name__  == "__main__":
    videoInfo = sv.VideoInfo.from_video_path(filePath)

    lineZone = sv.LineZone(start=lineTop, end=lineBottom)
    
    model = YOLO("yolo11l.pt")

    byteTrack = sv.ByteTrack(frame_rate=videoInfo.fps) 

    frameGenerator = sv.get_video_frames_generator(filePath)

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=videoInfo.resolution_wh)
    textScale = sv.calculate_optimal_text_scale(resolution_wh=videoInfo.resolution_wh)

    boundingBoxAnnotator = sv.BoxAnnotator(
        thickness=thickness, 
        color_lookup=sv.ColorLookup.TRACK)
    
    labelAnnotator = sv.LabelAnnotator(
        text_thickness=thickness,
        text_scale=textScale,
        color_lookup=sv.ColorLookup.TRACK
    )

    lineZoneAnnotator = sv.LineZoneAnnotator(
        thickness=thickness,
        text_scale=textScale
    )

    workerZoneOne = sv.PolygonZone(polygon=workerZoneOne)
    glassesZone = sv.PolygonZone(polygon=glassesZone, triggering_anchors=[sv.Position.TOP_CENTER]) #I needed to change the anchor point so it counts as 
                                                                                                   #In the zone if the top center is in it because there
                                                                                                   #are lots of time that it should be tracking the id but
                                                                                                   #since only the top half of the person is in the zone it
                                                                                                   #doesn't count it
    legitEntryZone = sv.PolygonZone(polygon=legitEntryZone)

    #uncomment if you want to see where the bounding boxes to the date/times are (also uncomment the matching stuff above/below)
    # dateZone = sv.PolygonZone(polygon=dateZone)
    # timeZone = sv.PolygonZone(polygon=timeZone)

    totalCustomers = set()
    crossedIn = set()
    crossedOut = set()
    glassesZoneInSeen = set()
    glassesZoneOutSeen = set()
    entryTimes = {}
    exitTimes = {}
    legitimateEntry = {}
    glassesZoneInDict = {}
    glassesZoneOutDict = {}
    firstSeenDict = {}
    lastSeenDict = {}

    
    with sv.VideoSink(target_path=targetPath, video_info=videoInfo) as sink:
        for frame in frameGenerator:

            result = model(frame, classes=0)[0]

            detections = sv.Detections.from_ultralytics(result)
            workerZoneOneTriggered = workerZoneOne.trigger(detections) # This returns as a numpy ndarray so we can use np.invert to only show the ones outside the box
            #if we were to print workerZoneOneTriggered it would look something like: [True False True False False]
            workerZoneOneTriggered = np.invert(workerZoneOneTriggered)
            detections = detections[workerZoneOneTriggered] #This keeps only the True ID's and discards the false ones
            detections = byteTrack.update_with_detections(detections=detections)

            
            #crossingIn/Out is a tuple with elements of people going in and out, it has 2 elements, both arrays such as (array([False, False]), array([False, True]))
            # so what I did was turn it into a list of only one element (for in we take element 0 and out is 1) then I took that element and did detections
            # on it so that only true value would appear and this solved my issue of figuring out who crossed the line correctly instead of just guessing
            
            crossingIn = lineZone.trigger(detections)
            crossingIn = list(crossingIn)
            crossingIn.pop(1)
            crossingIn = crossingIn[0]

            crossingIn = detections[crossingIn]

            crossingOut = lineZone.trigger(detections)
            crossingOut = list(crossingOut)
            crossingOut.pop(0)
            crossingOut = crossingOut[0]

            crossingOut = detections[crossingOut]

            #This stuff gets the time/date of the frame, and then we use that to say when someone crossed and at what time, idk if I will use date yet
            dateZoneCropped = frame[dateY: dateY + dateH, dateX: dateX + dateW]
            timeZoneCropped = frame[timeY: timeY + timeH, timeX: timeX + timeW]

            ocrDate = pytesseract.image_to_string(dateZoneCropped)
            ocrTime = pytesseract.image_to_string(timeZoneCropped)
            ocrTime = ocrTime.strip()

            #This ensures that the OCR time is in the correct HH:MM:SS format and not in a messed up format
            match = re.match(r"(\d{2}):(\d{2}):(\d{2})", ocrTime)
            if match:
                ocrTime = ":".join(match.groups())
            else:
                ocrTime = "N/A"

            # adds every seens ID into a set so that we can use it later in the CSV and also adds a lastseen dict so if an id disappears before leaving
            #we will take that lastseen time as their exit time. It also adds a firstseen dict so that even if someone is not a legit entry
            #we can still estimate their total time spent
            for tracker_id in detections.tracker_id:
                tracker_id = int(tracker_id) #we do this because it is a np.int64, but it is easier to look at/ work with if it is a regular integer
                lastSeenDict[tracker_id] = ocrTime
                totalCustomers.add(tracker_id)

                if tracker_id not in firstSeenDict:
                    firstSeenDict[tracker_id] = ocrTime

            legitEntry = legitEntryZone.trigger(detections)
            legitEntry = detections[legitEntry]

            for tracker_id in legitEntry.tracker_id:
                tracker_id = int(tracker_id)
                if tracker_id in crossedIn:
                    legitimateEntry[tracker_id] = True
                    print(f"{tracker_id} entered legitimately")
                else:
                    legitimateEntry.setdefault(tracker_id, False)
                    print(f"{tracker_id} entered  NOT legitimately")
                    

            for tracker_id in crossingIn.tracker_id:
                tracker_id = int(tracker_id)
                if tracker_id not in crossedIn:
                    print(f"ID {tracker_id} entered the store at {ocrTime}")
                    entryTimes[tracker_id] = ocrTime
                    crossedIn.add(tracker_id)

            for tracker_id in crossingOut.tracker_id:
                tracker_id = int(tracker_id)
                if tracker_id not in crossedOut:
                    print(f"ID {tracker_id} left the store at {ocrTime}")
                    exitTimes[tracker_id] = ocrTime
                    crossedOut.add(tracker_id)

            # For getting the time that someone enters the glassesZone
            glassesZoneIn = glassesZone.trigger(detections)
            glassesZoneIn = detections[glassesZoneIn]

            for tracker_id in glassesZoneIn.tracker_id:
                tracker_id = int(tracker_id)
                if tracker_id not in glassesZoneInSeen:
                    glassesZoneInDict[tracker_id] = ocrTime
                    glassesZoneInSeen.add(tracker_id)

            # For getting the time that someone exits the glassesZone

            glassesZoneOut = glassesZone.trigger(detections)
            glassesZoneOut = np.invert(glassesZoneOut)
            glassesZoneOut = detections[glassesZoneOut]

            for tracker_id in glassesZoneOut.tracker_id:
                tracker_id = int(tracker_id)
                if tracker_id in glassesZoneInSeen and tracker_id not in glassesZoneOutSeen:
                    glassesZoneOutDict[tracker_id] = ocrTime
                    glassesZoneOutSeen.add(tracker_id)

            #detections.xyxy prints something like
            # [[     1075.6      622.46      1179.7      975.01]
            # [     333.28      535.39      431.53      778.53]
            # [     254.77      652.97      301.02      755.86]
            # [     384.67      701.37      478.85      786.96]
            # [     990.59      687.06      1058.6      763.17]]

            # If I loop through it, then each element will be a different person tracked and then I can crop the frame around them and it will be easier

            for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
                x1, y1, x2, y2 = map(int, xyxy)
                personCropped = frame[y1:y2, x1:x2] #This crops the frame to just show whichever id it is at, we will then
                # cv.imwrite(f"cropped_person_{int(tracker_id)}.jpg", personCropped)
                

            labels = [
                f"# {tracker_id} {conf:.2f}"
                for tracker_id, conf
                in zip(detections.tracker_id, detections.confidence)
            ]

            annotatedFrame = frame.copy()
            annotatedFrame = boundingBoxAnnotator.annotate(
                scene=annotatedFrame, detections=detections
            )

            annotatedFrame = sv.draw_line(annotatedFrame, start=lineTop, end=lineBottom, color=sv.Color.RED, thickness=thickness)
            annotatedFrame = sv.draw_polygon(annotatedFrame, polygon=workerZoneOne.polygon, color=sv.Color.RED)
            annotatedFrame = sv.draw_polygon(annotatedFrame, polygon=glassesZone.polygon, color=sv.Color.GREEN)
            annotatedFrame = sv.draw_polygon(annotatedFrame, polygon=legitEntryZone.polygon, color = sv.Color.BLACK)

            #uncomment if you want to see where the bounding boxes to the date/times are
            # annotatedFrame = sv.draw_polygon(annotatedFrame, polygon=dateZone.polygon, color=sv.Color.BLUE)
            # annotatedFrame = sv.draw_polygon(annotatedFrame, polygon=timeZone.polygon, color=sv.Color.BLACK)

            annotatedFrame = lineZoneAnnotator.annotate(annotatedFrame, line_counter=lineZone)

            annotatedFrame = labelAnnotator.annotate(
                scene=annotatedFrame, detections=detections, labels=labels
            )

            sink.write_frame(frame=annotatedFrame) #This adds the frame so we can watch the output as a video

            cv.imshow("annotated_frame", annotatedFrame)
            if cv.waitKey(1) == ord("q"):
                break

        cv.destroyAllWindows()

        for tracker_id in totalCustomers:
            if tracker_id not in entryTimes:
                entryTimes[tracker_id] = firstSeenDict.get(tracker_id, "N/A")
        
            if tracker_id not in exitTimes:
                exitTimes[tracker_id] = lastSeenDict.get(tracker_id, "N/A")
            if tracker_id not in glassesZoneOutDict:
                glassesZoneOutDict[tracker_id] = lastSeenDict.get(tracker_id, "N/A")
        
        totalDuration, totalDurationSeconds = totalTimeCalc(entryTimes, exitTimes)
        glassesZoneDuration, totalGlassesZoneDurationSeconds = totalTimeCalc(glassesZoneInDict, glassesZoneOutDict)

        print(glassesZoneDuration)

        validGlassesZoneDuration = {}

        for tracker_id, time in glassesZoneDuration.items():
            if time != 'N/A':
                validGlassesZoneDuration[tracker_id] = time

        print(validGlassesZoneDuration)

        averageTotalDuration = totalDurationSeconds / len(totalCustomers)
        averageGlassesZoneDuration = totalGlassesZoneDurationSeconds / len(validGlassesZoneDuration)


        print(averageTotalDuration)
        print(averageGlassesZoneDuration)

        totalCustomers = list(totalCustomers)

        headers = ["ID", "Entry Time", "Exit Time", "Total Time in Store", "Total Time Browsing Glasses", "Legitimate Entrance"]


        with open("customer_log.csv", 'w', newline="") as customerLog:
            csvWriter = csv.writer(customerLog)
            csvWriter.writerow(headers)
                    
            for tracker_id in totalCustomers:

                entry = entryTimes.get(tracker_id, "N/A")
                exit = exitTimes.get(tracker_id, "N/A")
                legit = legitimateEntry.get(tracker_id, "N/A")
                totalCustomerDuration = totalDuration.get(tracker_id, "N/A")
                totalGlassesZoneDuration = glassesZoneDuration.get(tracker_id, "N/A")


                csvWriter.writerow([tracker_id, entry, exit, totalCustomerDuration, totalGlassesZoneDuration, legit])

#After doing some more stuff, I need to add testing/edge case stuff

#To do face detection I think I can do something semi-similar to what I did with the time but slightly different approach, I think I can cut the frame
# around each ID (or just one for now) and then I can use a face tracker somehow, then with the face tracker I will save the face as a numpy array or
# something. Then I will log it in a csv or some file to refer back to later.
