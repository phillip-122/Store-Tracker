import logging
from datetime import datetime

def finalize_id(tracker_id, reid_id_map):
    """
    Convert tracker_id to final_id using the reid_id_map so all the IDs are correct
    and people do not get duplicate IDs attached to them. 
    EX: If ID 6 should be 3 this turns it into 3 using the ReID map

    Parameters:
    tracker_id: np.int64 (but transformed into an int)
        This is the original tracker id.
    reid_id_map: dict
        This is a dictionary mapping tracker IDs to their matched ReID IDs.

    Returns:
    final_id: int
        This is the final ID after applying the ReID mapping.
    """
    tracker_id = int(tracker_id) #we do this because it is a np.int64, but it is easier to look at/ work with if it is a regular integer
    final_id = reid_id_map.get(tracker_id, tracker_id)

    return final_id

def secondsToString(seconds):
    """
    Convert a number of seconds into a formatted string with hours, minutes, and seconds.

    Parameters:
    seconds: int
        This is the total number of seconds to convert.

    Returns:
    secondsString: str
        This is a string in the format "<hours> hours <minutes> minutes <seconds> seconds".
    minutes: int
        This is an integer representing the minutes only.
    """

    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    secondsString = (f"{hours} hours {minutes} minutes {seconds} seconds")

    return secondsString, minutes

def totalTimeCalc(entryTime, exitTime):
    """
    Calculate the total duration between entry and exit times for each tracked ID.

    Parameters:
    entryTime: dict
        This is a dictionary mapping customer IDs to entry times in HH:MM:SS format.
    exitTime: dict
        This is a dictionary mapping customer IDs to exit times in HH:MM:SS format.

    Returns:
    totalDuration: dict
        This is a dictionary mapping IDs to formatted total durations.
    totalDurationSeconds: int
        This is the sum of all durations in seconds.
    entryTimeHours: list[int]
        This is a list of entry hours for each ID.
    durationMinutes: list[int]
        This is a list of durations in minutes for each ID.
    """

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
                logging.warning(f"Error getting times for ID {tracker_id}: {e}")
        else:
            totalDuration[tracker_id] = "N/A"

    return totalDuration, totalDurationSeconds, entryTimeHours, durationMinutes

