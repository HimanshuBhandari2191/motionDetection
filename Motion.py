import cv2
import pandas as pd
from datetime import datetime

static_back = None
motion_list = [None, None]
time = []

df = pd.DataFrame(columns=["Start", "End"])

# Background Subtractor
back_sub = cv2.createBackgroundSubtractorMOG2()

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    motion = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if static_back is None:
        static_back = gray
        continue

    fg_mask = back_sub.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 5000:  # Adjusted contour area threshold
            continue
        motion = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    motion_list.append(motion)
    motion_list = motion_list[-2:]

    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(datetime.now())

    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(datetime.now())

    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        if motion == 1:
            time.append(datetime.now())
        break

time_data = [{"Start": time[i], "End": time[i + 1]} for i in range(0, len(time), 2)]
time_df = pd.DataFrame(time_data)
df = pd.concat([df, time_df], ignore_index=True)

df.to_csv("Time_of_movements.csv", index=False)

video.release()
cv2.destroyAllWindows()
