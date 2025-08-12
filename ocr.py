import re
import pytesseract
from config import TESSERACT_PATH

pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)

def getTime(timeZoneCropped, lastProperFormatedTime):
    ocrTime = pytesseract.image_to_string(timeZoneCropped)
    ocrTime = ocrTime.strip()
 
    #I may add something that says that if it is not right time format, it just uses the last proper formatted time
    #This ensures that the OCR time is in the correct HH:MM:SS format and not in a messed up format
    match = re.search(r"(\d{2}):(\d{2}):(\d{2})", ocrTime)
    if match:
        ocrTime = ":".join(match.groups())
        lastProperFormatedTime = ocrTime
    else:
        ocrTime = lastProperFormatedTime

    return ocrTime, lastProperFormatedTime
