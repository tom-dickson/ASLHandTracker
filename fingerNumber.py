import cv2 #type: ignore
import time
import handDetector as hd

cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0
tipIds = [4, 8, 12, 16, 20]
detector = hd.handDetector(maxHands=4, detectionCon=0.75)

while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    totalFingers = 0

    if len(lmList) != 0:
        fingers = []

        #thumb right hand
        if lmList[4][1] > lmList[3][1]: fingers.append(1)
        else: fingers.append(0)

        #fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]: fingers.append(1)
            else: fingers.append(0)
        totalFingers = sum(fingers)

    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f'FPS: {fps}', (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(frame, f'Num Fingers: {totalFingers}', (50, h-100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break