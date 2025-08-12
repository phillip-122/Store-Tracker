import cv2 as cv
import torch
from utils import finalize_id

def getCrossing(lineZone, detections, index):
    crossing = list(lineZone.trigger(detections))
    crossing.pop(index) #we pop the index of the part of the list we don't want
                        #It returns a list like [[True False], [False False]], where index 0 is cross ins and index 1 is cross outs so we pop the opposite
    crossing = detections[crossing[0]]

    return crossing

def reid(detections, frame, extractor, reidSet, reidDict, reid_id_map, REID_THRESHOLD):
    original_ids = []
    final_ids = []
            
    for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
        tracker_id = int(tracker_id)
        original_ids.append(tracker_id)
                
        bestScore = 0.0
        bestID = None

        x1, y1, x2, y2 = map(int, xyxy)
        personCropped = frame[y1:y2, x1:x2] #This crops the frame to just show whichever id it is at, we will then
        personCropped = cv.cvtColor(personCropped, cv.COLOR_BGR2RGB) #we need it to be in RGB, but by default it is in BGR

        feat = extractor(personCropped)

        if tracker_id not in reidSet:
            for ids, feats in reidDict.items():
                closestScore = max(torch.nn.functional.cosine_similarity(feat, f).item() for f in feats) #We loop through and find whichever score is highest and use that for comparison
            
                if closestScore > bestScore:
                    bestScore = closestScore
                    bestID = ids

            if bestScore > REID_THRESHOLD:
                # print(f"Thus, ID: {tracker_id} and ID: {bestID} are actually the same person")
                reid_id_map[tracker_id] = bestID
                final_id = bestID
            else:
                reid_id_map[tracker_id] = tracker_id
                final_id = tracker_id


            reidSet.add(tracker_id)
        else:
            final_id = finalize_id(tracker_id, reid_id_map)

            reidDict.setdefault(final_id, []).append(feat) #makes a dict of {tracker_id: [list of features for each id]}
            final_ids.append(final_id)
