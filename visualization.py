import supervision as sv

def annotateFrame(frame, detections, boundingBoxAnnotator, lineZoneAnnotator, labelAnnotator, labels, thickness, lineTop, lineBottom, workerZone, glassesZone, legitEntryZone, lineZone):
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

    return annotatedFrame