import logging
import supervision as sv

def annotateFrame(frame, detections, boundingBoxAnnotator, lineZoneAnnotator, labelAnnotator, labels, thickness, lineTop, lineBottom, workerZone, glassesZone, legitEntryZone, lineZone):
    """
    Annotate the frame with bounding boxes, labels, and polygon zone overlays.

    Parameters:
    frame: numpy.ndarray
        This is the current video frame to annotate.
    detections: sv.Detections
        This is the detected objects for the current frame.
    boundingBoxAnnotator: sv.BoxAnnotator
        This is the annotator for drawing bounding boxes.
    lineZoneAnnotator: sv.LineZoneAnnotator
        This is the annotator for visualizing the entrance/exit line.
    labelAnnotator: sv.LabelAnnotator
        This is the annotator for displaying labels on detections.
    labels: list[str]
        This is a list of label strings for each detection.
    thickness: int
        This is the thickness for drawing the bounding boxes and lines.
    lineTop: sv.Point
        This is the top point of the line zone.
    lineBottom: sv.Point
        This is the bottom point of the line zone.
    workerZone: sv.PolygonZone
        This is a polygon zone defining the worker area.
    glassesZone: sv.PolygonZone
        This is a polygon zone defining the glasses display area.
    legitEntryZone: sv.PolygonZone
        This is a polygon zone defining the legitimate entry area.
    lineZone: sv.LineZone
        This is the line zone object for entry/exit detection.

    Returns:
    annotatedFrame: numpy.ndarray
        This is the annotated frame ready for display or saving.
    """
    
    
    annotatedFrame = frame.copy()
    annotatedFrame = boundingBoxAnnotator.annotate(
        scene=annotatedFrame, detections=detections
    )

    annotatedFrame = sv.draw_line(annotatedFrame, start=lineTop, end=lineBottom, color=sv.Color.RED, thickness=thickness)
    annotatedFrame = sv.draw_polygon(annotatedFrame, polygon=workerZone.polygon, color=sv.Color.RED)
    annotatedFrame = sv.draw_polygon(annotatedFrame, polygon=glassesZone.polygon, color=sv.Color.GREEN)
    annotatedFrame = sv.draw_polygon(annotatedFrame, polygon=legitEntryZone.polygon, color = sv.Color.BLACK)

    #uncomment if you want to see where the bounding boxes to the times are
    # annotatedFrame = sv.draw_polygon(annotatedFrame, polygon=timeZone.polygon, color=sv.Color.BLUE)

    annotatedFrame = lineZoneAnnotator.annotate(annotatedFrame, line_counter=lineZone)

    annotatedFrame = labelAnnotator.annotate(
        scene=annotatedFrame, detections=detections, labels=labels
    )

    logging.debug("Annotated frame with %d detections.", len(detections))

    return annotatedFrame