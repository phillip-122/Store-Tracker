import argparse
from ultralytics import YOLO
import supervision as sv
import cv2 as cv
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# filePath = "C:\\Users\\PHILLIP\\Desktop\\Code Projects\\python\\Store-Tracker\\Videos - Copy\\merged_trimmed_20250523.mp4"
filePath = r"C:\Users\PHILLIP\Downloads\merged_trimmed_short - Made with Clipchamp.mp4"
targetPath = "C:\\Users\\PHILLIP\\Desktop\\Code Projects\\python\\Store-Tracker\\output.mp4"

lineTop = sv.Point(x=300, y=0) #make x 150
lineBottom = sv.Point(x=300, y=900)

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



def parse_arguments() -> argparse.Namespace:
    parsar = argparse.ArgumentParser(
        description="Store customer tracking"
    )
    parsar.add_argument(
        "--filePath",
        required=False,
        default=filePath,
        help="Path to the source video file",
        type=str,
    )
    return parsar.parse_args()

if __name__  == "__main__":
    args = parse_arguments()

    videoInfo = sv.VideoInfo.from_video_path(args.filePath)

    print(videoInfo.fps)
    print(videoInfo.resolution_wh) #1920x1080

    lineZone = sv.LineZone(start=lineTop, end=lineBottom)
    
    model = YOLO("yolo11l.pt")

    byteTrack = sv.ByteTrack(frame_rate=videoInfo.fps) 

    frameGenerator = sv.get_video_frames_generator(args.filePath)

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
    glassesZone = sv.PolygonZone(polygon=glassesZone)

    #uncomment if you want to see where the bounding boxes to the date/times are (also uncomment the matching stuff above/below)
    # dateZone = sv.PolygonZone(polygon=dateZone)
    # timeZone = sv.PolygonZone(polygon=timeZone)

    crossedIn = set()
    entryTimes = {}
    crossedOut = set()
    exitTimes = {}

    
    with sv.VideoSink(target_path=targetPath, video_info=videoInfo) as sink:
        for frame in frameGenerator:

            result = model(frame, classes=0)[0]

            detections = sv.Detections.from_ultralytics(result)
            workerZoneOneTriggered = workerZoneOne.trigger(detections) # This returns as a numpy ndarray so we can use np.invert to only show the ones outside the box
            workerZoneOneTriggered = np.invert(workerZoneOneTriggered)
            detections = detections[workerZoneOneTriggered]
            # detections = detections[glassesZone.trigger(detections)]
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

            for tracker_id in crossingIn.tracker_id:
                tracker_id = int(tracker_id)
                ocrTime = ocrTime.strip()
                if tracker_id not in crossedIn:
                    print(f"ID {tracker_id} entered the store at {ocrTime}")
                    entryTimes[tracker_id] = ocrTime
                    crossedIn.add(tracker_id)

            for tracker_id in crossingOut.tracker_id:
                tracker_id = int(tracker_id)
                ocrTime = ocrTime.strip()
                if tracker_id not in crossedOut:
                    print(f"ID {tracker_id} left the store at {ocrTime}")
                    exitTimes[tracker_id] = f"{ocrTime}"
                    crossedOut.add(tracker_id)

            print(entryTimes)
            print(exitTimes)
                

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

            #uncomment if you want to see where the bounding boxes to the date/times are
            # annotatedFrame = sv.draw_polygon(annotatedFrame, polygon=dateZone.polygon, color=sv.Color.BLUE)
            # annotatedFrame = sv.draw_polygon(annotatedFrame, polygon=timeZone.polygon, color=sv.Color.BLACK)

            annotatedFrame = lineZoneAnnotator.annotate(annotatedFrame, line_counter=lineZone)

            annotatedFrame = labelAnnotator.annotate(
                scene=annotatedFrame, detections=detections, labels=labels
            )

            sink.write_frame(frame=annotatedFrame)

            cv.imshow("annotated_frame", annotatedFrame)
            if cv.waitKey(1) == ord("q"):
                break


    cv.destroyAllWindows()


#To do face detection I think I can do something semi-similar to what I did with the time but slightly different approach, I think I can cut the frame
# around each ID (or just one for now) and then I can use a face tracker somehow, then with the face tracker I will save the face as a numpy array or
# something. Then I will log it in a csv or some file to refer back to later.

#I need to make a thing that 1. can extract the time from the screen, 2. say when a new ID appears say it appeared at that time, 3. When someone goes in/out
# it needs to say like ID 42 entered at 3:07:41


#maybe to bridge the info together I make a list or dictionary or something so that it says like 1. 0.8964 x1=612.8533 y1 = 767.7429 x2=934.3393 y2 = 986.6265
