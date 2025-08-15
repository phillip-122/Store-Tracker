import logging
import csv
import matplotlib.pyplot as plt
from collections import Counter
from store_tracker.utils import secondsToString, totalTimeCalc, finalize_id


def barChart(entryTimeHours):
    labels = ["9 am", "10 am", "11 am", "12 pm", "1 pm", "2 pm", "3 pm", "4 pm", "5 pm", "6 pm"]
    plt.style.use('ggplot')
    plt.figure()
    plt.bar(entryTimeHours.keys(), entryTimeHours.values(), edgecolor = 'black')
    plt.title("Peak Customer hours")
    plt.xlabel("Hours")
    plt.ylabel("Number of Customers")
    plt.xlim(8.5, 18.5) # The reason we have both xlim and xticks is because if an hour has 0 people, we still want to show it as that and not cut it off
    plt.xticks(range(9, 19), labels=labels)
    plt.tight_layout()
    plt.savefig("Peak_Customer_Hours.png")
    logging.info("Saved bar chart as Peak_Customer_Hours.png")

def histogram(durationMinutes):
    plt.figure()
    plt.hist(durationMinutes, bins=range(0, 70, 10), edgecolor = 'black')
    plt.title('Amount of time Customers Stayed')
    plt.xlabel('Customer Visit Duration')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    plt.savefig("Duration_Customer_Visit.png")
    logging.info("Saved histogram as Duration_Customer_Visit.png")

def saveCSV(CSV_OUTPUT, totalCustomers, entryTimes, exitTimes, legitimateEntry, totalDuration, glassesZoneDuration, reid_id_map):
    """
    Save customer tracking data to a CSV file.

    Parameters:
    CSV_OUTPUT: pathlib.WindowsPath
        This is the output path for the CSV
    totalCustomers: list
        This is the list of all tracked customer IDs.
    entryTimes: dict
        This is a dictionary mapping customer IDs to entry times.
    exitTimes: dict
        This is a dictionary mapping customer IDs to exit times.
    legitimateEntry: dict
        This is a dictionary mapping customer IDs to whether they entered legitimately.
    totalDuration: dict
        This is a dictionary mapping customer IDs to total time spent in the store.
    glassesZoneDuration: dict
        This is a dictionary mapping customer IDs to total time spent in the glasses zone.
    reid_id_map: dict
        This is a dictionary mapping tracker IDs to their matched ReID IDs.

    Returns:
    None
        This function does not return anything. It writes the data to a CSV file.
    """

    totalCustomers = list(totalCustomers)

    headers = ["ID", "Entry Time", "Exit Time", "Total Time in Store", "Total Time Browsing Glasses", "Legitimate Entrance"]


    with open(CSV_OUTPUT, 'w', newline="") as customerLog:
        csvWriter = csv.writer(customerLog)
        csvWriter.writerow(headers)
                
        for tracker_id in totalCustomers:
            final_id = finalize_id(tracker_id, reid_id_map)

            entry = entryTimes.get(final_id, "N/A")
            exit = exitTimes.get(final_id, "N/A")
            legit = legitimateEntry.get(final_id, "N/A")
            totalCustomerDuration = totalDuration.get(final_id, "N/A")
            totalGlassesZoneDuration = glassesZoneDuration.get(final_id, "N/A")


            csvWriter.writerow([final_id, entry, exit, totalCustomerDuration, totalGlassesZoneDuration, legit])
    logging.info("Saved CSV to %s", CSV_OUTPUT)

def saveStats(CSV_OUTPUT, totalCustomers, entryTimes, exitTimes, firstSeenDict, lastSeenDict, glassesZoneInDict, glassesZoneOutDict, legitimateEntry, reid_id_map):
    """
    Generate analytics, calculate durations, plot charts, and save customer CSV data.

    Parameters:
    CSV_OUTPUT: pathlib.WindowsPath
        This is the output path for the CSV
    totalCustomers: set
        This is the set of all tracked customer IDs.
    entryTimes: dict
        This is a dictionary mapping customer IDs to entry times.
    exitTimes: dict
        This is a dictionary mapping customer IDs to exit times.
    firstSeenDict: dict
        This is a dictionary mapping customer IDs to first detection time.
    lastSeenDict: dict
        This is a dictionary mapping customer IDs to last detection time.
    glassesZoneInDict: dict
        This is a dictionary mapping customer IDs to glasses zone entry times.
    glassesZoneOutDict: dict
        This is a dictionary mapping customer IDs to glasses zone exit times.
    legitimateEntry: dict
        This is a dictionary mapping customer IDs to whether they entered legitimately.
    reid_id_map: dict
        This is a dictionary mapping tracker IDs to their matched ReID IDs.

    Returns:
    None
        This function does not return anything. It generates plots and saves CSV output.
    """

    for tracker_id in totalCustomers:
        final_id = finalize_id(tracker_id, reid_id_map)

        if final_id not in entryTimes:
            entryTimes[final_id] = firstSeenDict.get(final_id, "N/A")
        if final_id not in exitTimes:
            exitTimes[final_id] = lastSeenDict.get(final_id, "N/A")
        if final_id not in glassesZoneOutDict:
            glassesZoneOutDict[final_id] = lastSeenDict.get(final_id, "N/A")
    
    totalDuration, totalDurationSeconds, entryTimeHours, durationMinutes = totalTimeCalc(entryTimes, exitTimes)
    glassesZoneDuration, totalGlassesZoneDurationSeconds, _, _ = totalTimeCalc(glassesZoneInDict, glassesZoneOutDict)

    barChart(Counter(entryTimeHours))
    histogram(Counter(durationMinutes))

    validGlassesZoneDuration = {}

    for tracker_id, duration in glassesZoneDuration.items():
        if duration != 'N/A':
            validGlassesZoneDuration[tracker_id] = duration

    try: #this is incase 0 customers are in the store/in glasses zone at the time you look at
        averageTotalDuration = totalDurationSeconds / len(totalCustomers)
        averageGlassesZoneDuration = totalGlassesZoneDurationSeconds / len(validGlassesZoneDuration)
    except Exception as e:
        logging.error(f"Error getting average duration: {e}")
        averageTotalDuration = averageGlassesZoneDuration = 0

    averageTotalDuration, _ = secondsToString(averageTotalDuration)
    averageGlassesZoneDuration, _ = secondsToString(averageGlassesZoneDuration)

    logging.info(f"Average time spent in store: {averageTotalDuration}")
    logging.info(f"Average time spent browsing glasses: {averageGlassesZoneDuration}")
    
    saveCSV(CSV_OUTPUT, totalCustomers, entryTimes, exitTimes, legitimateEntry, totalDuration, glassesZoneDuration, reid_id_map)