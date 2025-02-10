import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os

# Initialize deques with consistent maxlen
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0

kernel = np.ones((5, 5), np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
colorIndex = 0

paintWindow = np.zeros((471, 636, 3)) + 255

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
ret = True
drawing = False

def clearCanvas():
    global bpoints, gpoints, rpoints, blue_index, green_index, red_index, paintWindow
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    blue_index = 0
    green_index = 0
    red_index = 0
    paintWindow = np.zeros((471, 636, 3)) + 255

while ret:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # UI Elements
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (520, 1), (615, 65), (0, 0, 0), 2)  # Fixed Y-coordinates
    
    # Button labels
    cv2.putText(frame, "CLOSE", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, "CLEAR", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])

        if (thumb[1] - center[1] < 30):
            bpoints.append(deque(maxlen=1024))  # Fixed maxlen consistency
            blue_index += 1
            gpoints.append(deque(maxlen=1024))
            green_index += 1
            rpoints.append(deque(maxlen=1024))
            red_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140:
                cap.release()
                cv2.destroyAllWindows()
                import project_selection
                project_selection.run()
            elif 160 <= center[0] <= 255:
                colorIndex = 0
            elif 275 <= center[0] <= 370:
                colorIndex = 1
            elif 390 <= center[0] <= 485:
                colorIndex = 2
            elif 520 <= center[0] <= 615:  # Fixed clear button check
                clearCanvas()
        elif drawing:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)

    points = [bpoints, gpoints, rpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):
        drawing = not drawing
    elif key == ord('o'):
        save_path = os.path.join('E:/aircanvas/images/', 'paint_drawing.png')
        cv2.imwrite(save_path, paintWindow)

cap.release()
cv2.destroyAllWindows()