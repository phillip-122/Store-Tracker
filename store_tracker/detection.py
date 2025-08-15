import logging
import cv2 as cv
import torch
from store_tracker.utils import finalize_id

def getCrossing(lineZone, detections, index):
    """
    Filter detections that crossed a given line in the specified direction.

    Parameters:
    lineZone: sv.LineZone
        This is the line zone object for detecting crossings.
    detections: sv.Detections
        This is the detected objects for the current frame.
    index: int
        This is the index that we pop off (0 for out, 1 for in).

    Returns:
    crossing: sv.Detections
        This is the set of detections that crossed the line in the specified direction.
    """

    crossing = list(lineZone.trigger(detections))
    crossing.pop(index) #we pop the index of the part of the list we don't want
                        #It returns a list like [[True False], [False False]], where index 0 is cross ins and index 1 is cross outs so we pop the opposite
    crossing = detections[crossing[0]]
    logging.debug(f"Detected {len(crossing)} crossings for {index} %d")

    return crossing

def reid(detections, frame, extractor, reidSet, reidDict, reid_id_map, REID_THRESHOLD):
    """
    Performs ReID (person re-identification) on detected people to maintain consistent IDs.

    Parameters:
    detections: sv.Detections
        This is the detected people in the current frame.
    frame: numpy.ndarray
        This is the current video frame.
    extractor: torchreid.utils.FeatureExtractor
        This is the model used to extract features for ReID.
    reidSet: set
        This is the set of tracker IDs that have been processed for ReID.
    reidDict: dict
        This is a dictionary mapping IDs to their extracted feature vectors.
    reid_id_map: dict
        This is a dictionary mapping tracker IDs to their matched ReID IDs.
    REID_THRESHOLD: float
        This is the similarity threshold for matching IDs.

    Returns:
    None
        This function does not return anything. It performs ReID on the IDs
    """

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
                logging.debug(f"Thus, ID: {tracker_id} and ID: {bestID} are actually the same person")
                reid_id_map[tracker_id] = bestID
                final_id = bestID
                logging.info(f"ReID matched tracker {tracker_id} to tracker {bestID} with score {bestScore:.3f}")
            else:
                reid_id_map[tracker_id] = tracker_id
                final_id = tracker_id
                logging.debug(f"No strong ReID match for tracker {tracker_id} (score {bestScore:.3f})")

            reidSet.add(tracker_id)
        else:
            final_id = finalize_id(tracker_id, reid_id_map)

            reidDict.setdefault(final_id, []).append(feat) #makes a dict of {tracker_id: [list of features for each id]}
            final_ids.append(final_id)
