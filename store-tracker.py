import cv2 as cv
import numpy as np
import supervision as sv

from config import *
from detection import getCrossing, reid
from analytics import stats
from visualization import annotateFrame
from utils import finalize_id
from ocr import getTime


def main():
    videoInfo = sv.VideoInfo.from_video_path(FILE_PATH)

    lineZone = sv.LineZone(start=LINE_TOP, end=LINE_BOTTOM)
    
    byteTrack = sv.ByteTrack(frame_rate=videoInfo.fps) 

    frameGenerator = sv.get_video_frames_generator(FILE_PATH)

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

    workerZone = sv.PolygonZone(polygon=WORKER_ZONE)
    glassesZone = sv.PolygonZone(polygon=GLASSES_ZONE, triggering_anchors=[sv.Position.TOP_CENTER]) #I needed to change the anchor point so it counts as 
                                                                                                   #In the zone if the top center is in it because there
                                                                                                   #are lots of time that it should be tracking the id but
                                                                                                   #since only the top half of the person is in the zone it
                                                                                                   #doesn't count it
    legitEntryZone = sv.PolygonZone(polygon=LEGIT_ENTRY_ZONE)

    #uncomment if you want to see where the bounding boxes to the date/times are (also uncomment the matching stuff above/below)
    # timeZone = sv.PolygonZone(polygon=timeZone)

    totalCustomers = set()
    crossedIn = set()
    crossedOut = set()
    glassesZoneInSeen = set()
    glassesZoneOutSeen = set()
    reidSet = set()

    entryTimes = {}
    exitTimes = {}
    legitimateEntry = {}
    glassesZoneInDict = {}
    glassesZoneOutDict = {}
    firstSeenDict = {}
    lastSeenDict = {}
    reidDict = {}
    reid_id_map = {}

    lastProperFormatedTime = "00:00:00" #This gets updated so we cannot put it in config

    with sv.VideoSink(target_path=TARGET_PATH, video_info=videoInfo) as sink:
        for frame in frameGenerator:

            result = model(frame, classes=0)[0]

            detections = sv.Detections.from_ultralytics(result)
            workerZoneTriggered = workerZone.trigger(detections) # This returns as a numpy ndarray so we can use np.invert to only show the ones outside the box
            #if we were to print workerZoneTriggered it would look something like: [True False True False False]
            workerZoneTriggered = np.invert(workerZoneTriggered)
            detections = detections[workerZoneTriggered] #This keeps only the True ID's and discards the false ones
            detections = byteTrack.update_with_detections(detections=detections)

            
            #crossingIn/Out is a tuple with elements of people going in and out, it has 2 elements, both arrays such as (array([False, False]), array([False, True]))
            # so what I did was turn it into a list of only one element (for in we take element 0 and out is 1) then I took that element and did detections
            # on it so that only true value would appear and this solved my issue of figuring out who crossed the line correctly instead of just guessing
            
            crossingIn = getCrossing(lineZone, detections, 1)
            crossingOut = getCrossing(lineZone, detections, 0)

            #This stuff gets the time of the frame, and then we use that to say when someone crossed and at what time
            timeZoneCropped = frame[TIME_Y: TIME_Y + TIME_H, TIME_X: TIME_X + TIME_W]

            print(lastProperFormatedTime)
            
            ocrTime, lastProperFormatedTime = getTime(timeZoneCropped, lastProperFormatedTime)

            reid(detections, frame, extractor, reidSet, reidDict, reid_id_map, REID_THRESHOLD)

            # adds every seens ID into a set so that we can use it later in the CSV and also adds a lastseen dict so if an id disappears before leaving
            #we will take that lastseen time as their exit time. It also adds a firstseen dict so that even if someone is not a legit entry
            #we can still estimate their total time spent
            for tracker_id in detections.tracker_id:
                final_id = finalize_id(tracker_id, reid_id_map)
                
                lastSeenDict[final_id] = ocrTime
                totalCustomers.add(final_id)

                if final_id not in firstSeenDict:
                    firstSeenDict[final_id] = ocrTime

            legitEntry = legitEntryZone.trigger(detections)
            legitEntry = detections[legitEntry]

            for tracker_id in legitEntry.tracker_id:
                final_id = finalize_id(tracker_id, reid_id_map)

                if final_id in crossedIn:
                    legitimateEntry[final_id] = True
                    # print(f"{final_id} entered legitimately")
                else:
                    legitimateEntry.setdefault(final_id, False)
                    # print(f"{final_id} entered  NOT legitimately")
                    

            for tracker_id in crossingIn.tracker_id:
                final_id = finalize_id(tracker_id, reid_id_map)

                if final_id not in crossedIn:
                    # print(f"ID {final_id} entered the store at {ocrTime}")
                    entryTimes[final_id] = ocrTime
                    crossedIn.add(final_id)

            for tracker_id in crossingOut.tracker_id:
                final_id = finalize_id(tracker_id, reid_id_map)

                if final_id not in crossedOut:
                    # print(f"ID {final_id} left the store at {ocrTime}")
                    exitTimes[final_id] = ocrTime
                    crossedOut.add(final_id)

            # For getting the time that someone enters the glassesZone
            glassesZoneIn = glassesZone.trigger(detections)
            glassesZoneIn = detections[glassesZoneIn]

            for tracker_id in glassesZoneIn.tracker_id:
                final_id = finalize_id(tracker_id, reid_id_map)

                if final_id not in glassesZoneInSeen:
                    glassesZoneInDict[final_id] = ocrTime
                    glassesZoneInSeen.add(final_id)

            # For getting the time that someone exits the glassesZone

            glassesZoneOut = glassesZone.trigger(detections)
            glassesZoneOut = np.invert(glassesZoneOut)
            glassesZoneOut = detections[glassesZoneOut]

            for tracker_id in glassesZoneOut.tracker_id:
                final_id = finalize_id(tracker_id, reid_id_map)

                if final_id in glassesZoneInSeen and final_id not in glassesZoneOutSeen:
                    glassesZoneOutDict[final_id] = ocrTime
                    glassesZoneOutSeen.add(final_id)

            labels = [
                f"# {reid_id_map.get(int(tracker_id), int(tracker_id))} {conf:.2f}"
                for tracker_id, conf in zip(detections.tracker_id, detections.confidence)
            ]

            annotatedFrame = annotateFrame(frame, detections, boundingBoxAnnotator, lineZoneAnnotator, labelAnnotator, 
                                           labels, thickness, LINE_TOP, LINE_BOTTOM, 
                                           workerZone, glassesZone, legitEntryZone, lineZone)

            sink.write_frame(frame=annotatedFrame) #This adds the frame so we can watch the output as a video

            cv.imshow("annotated_frame", annotatedFrame)
            if cv.waitKey(1) == ord("q"):
                break

        cv.destroyAllWindows()

        stats(CSV_OUTPUT, totalCustomers, entryTimes, exitTimes, firstSeenDict, lastSeenDict, glassesZoneInDict, glassesZoneOutDict, legitimateEntry, reid_id_map)

        
if __name__  == "__main__":
    main()