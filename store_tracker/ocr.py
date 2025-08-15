import logging
import re
import pytesseract
from store_tracker.config import TESSERACT_PATH

pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)

def getTime(timeZoneCropped, lastProperFormatedTime):
    """
    Perform OCR on the cropped frame region containing the timestamp and return the detected time.

    Falls back to the last valid timestamp if OCR fails to detect a proper "HH:MM:SS" format.

    Parameters:
    timeZoneCropped: numpy.ndarray
        This is the cropped frame region that contains the time.
    lastProperFormatedTime: str
        This is the last valid timestamp to fall back on if OCR fails.

    Returns:
    ocrTime: str
        The extracted timestamp (or the last valid timestamp if OCR fails).
    lastProperFormatedTime: str
        The updated last valid timestamp.
    """

    ocrTime = pytesseract.image_to_string(timeZoneCropped)
    ocrTime = ocrTime.strip()
 
    #This ensures that the OCR time is in the correct HH:MM:SS format and not in a messed up format
    match = re.search(r"(\d{2}):(\d{2}):(\d{2})", ocrTime)
    if match:
        ocrTime = ":".join(match.groups())
        lastProperFormatedTime = ocrTime
        logging.debug(f"OCR extracted time: {ocrTime}")
    else:
        if ocrTime != lastProperFormatedTime:  #This will only log if lastProperFormatedTime is a new value so that tracker.log isn't filled with repeat lines
            logging.warning(f"OCR failed, using last valid time: {lastProperFormatedTime}")
        ocrTime = lastProperFormatedTime

    return ocrTime, lastProperFormatedTime
