import cv2
import os
import numpy as np
import pygame.mixer as mixer
from tensorflow.keras.models import load_model


# importing the alarm sound to be used to alert the drowsy driver
mixer.init()
sound = mixer.Sound("alarm.wav")
isPlaying = False


# importing the har cascades files
face = cv2.CascadeClassifier("cascade_files/haarcascade_frontalface_alt.xml")
leye = cv2.CascadeClassifier("cascade_files/haarcascade_lefteye_2splits.xml")
reye = cv2.CascadeClassifier("cascade_files/haarcascade_righteye_2splits.xml")

lbl = ["Close", "Open"]


# importing the trained model
model = load_model("models/drowsiness_model.h5")


path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 16
rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(
        gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25)
    )
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(
        frame, (100, height + 50), (200, height), (0, 0, 0), thickness=cv2.FILLED
    )

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for x, y, w, h in right_eye:
        r_eye = frame[y : y + h, x : x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)

        predictions = model.predict(r_eye)
        threshold = 0.5  # Example threshold for binary classification
        rpred = (predictions > threshold).astype(int)

        if rpred[0] == 1:
            lbl = "Open"
        if rpred[0] == 0:
            lbl = "Closed"
        break
    for x, y, w, h in left_eye:
        l_eye = frame[y : y + h, x : x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)

        predictions = model.predict(l_eye)
        threshold = 0.5  # Example threshold for binary classification
        lpred = (predictions > threshold).astype(int)

        if lpred[0] == 1:
            lbl = "Open"
        if lpred[0] == 0:
            lbl = "Closed"
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score = score + 1
        cv2.putText(
            frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA
        )

    else:
        score = score - 1
        cv2.putText(
            frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA
        )

    if score < 0:
        score = 0
    cv2.putText(
        frame,
        "Score:" + str(score),
        (100, height - 20),
        font,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if score > 15:
        if score > 19:
            score = 19
        # person is feeling sleepy so we beep the alarm

        if not isPlaying:
            sound.play(loops=-1)
            isPlaying = True

        if thicc == 16:
            thicc = 8
        else:
            thicc = 16

        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    else:
        isPlaying = False
        sound.stop()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()

cv2.destroyAllWindows()
