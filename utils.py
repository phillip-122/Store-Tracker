from datetime import datetime

def finalize_id(tracker_id, reid_id_map):
    """
    Convert tracker_id to final_id using reid_id_map
    
    """
    tracker_id = int(tracker_id) #we do this because it is a np.int64, but it is easier to look at/ work with if it is a regular integer
    final_id = reid_id_map.get(tracker_id, tracker_id)

    return final_id

def secondsToString(seconds):

    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    secondsString = (f"{hours} hours {minutes} minutes {seconds} seconds")

    return secondsString, minutes

def totalTimeCalc(entryTime, exitTime):
    totalDuration = {}
    totalDurationSeconds = 0
    entryTimeHours = []
    durationMinutes = []

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

                entryTimeHours.append(entry.hour)

                #This is so I can format the output so it says _ Hours _ Minutes _ Seconds
                totalSeconds = int(duration.total_seconds())
                secondsString, minutes = secondsToString(totalSeconds)

                totalDuration[tracker_id] = secondsString

                durationMinutes.append(minutes)

                totalDurationSeconds += totalSeconds
                
            except Exception as e:
                print(f"Error getting times for ID {tracker_id}: {e}")
        else:
            totalDuration[tracker_id] = "N/A"

    return totalDuration, totalDurationSeconds, entryTimeHours, durationMinutes

