import cv2
from handDetector import handDetector as hd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier as dtc

def preprocess(lmlist):
    # should do something to account for the hand location being in different places like on nn.py
    # have like this for now because I am trying different ways that might work better
    return lmlist

data = np.load("dataNew.npy")
targets = np.load("labelsNew.npy")
classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
           "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

data = data.reshape(data.shape[0], 21*3)
preprocessedData = np.zeros_like(data)
for idx, datum in enumerate(data):
    preprocessedData[idx] = preprocess(datum)
data = preprocessedData.copy()
tree = dtc()
tree = tree.fit(data, targets)
print("CVS", cross_val_score(tree, data, targets))


cap = cv2.VideoCapture(0)
detector = hd(maxHands=1)
while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame)
    letter = "nothing"
    
    if lmlist:
        preprocessedHand = preprocess(np.array(lmlist, dtype=float).reshape(1, 21*3))
        pred = tree.predict(preprocessedHand)[0]
        letter = classes[pred]

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, letter, (h-20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break
