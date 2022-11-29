from handDetector import handDetector as hd
import cv2
import time
from sklearn.neural_network import MLPClassifier as MLPC
import numpy as np

'''
This file was to try out something different, which was implemented in nn.py
Also tried MLPClassifier from sklearn, but just don't get the same results

Feel free to mess around here, its basically dead or we can delete it
'''

labels = np.load('labelsNew.npy') # 19436 labels
data = np.load('dataNew.npy').astype(float)    # 19436 preprocessed images 
                               # of x,y,z at 21 landmarks a hand

data = data.reshape(data.shape[0], 21*3) # including z

def preprocess(lmlist):
  max = lmlist.max(axis=0)
  min = lmlist.min(axis=0)
  newList = np.zeros_like(lmlist)

  if max[0] - min[0] == 0: max[0] += 0.0001
  if max[1] - min[1] == 0: max[1] += 0.0001
  if max[2] - min[2] == 0: max[2] += 0.0001
  newList[:, 0] = (lmlist[:, 0] - min[0]) / (1.0 * max[0] - min[0])
  newList[:, 1] = (lmlist[:, 1] - min[1]) / (1.0 * max[1] - min[1])
  newList[:, 2] = (lmlist[:, 2] - min[2]) / (1.0 * max[2] - min[2])
  return newList

translate = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", 
            "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
    ]

newTranslate = ["A", "B", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 
             "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "nothing"
    ]

removed = [2, 3, 14, 15, 16, 26, 28]

for r in removed:
    labels = np.where(labels == r, 27, labels)

for i in range(2, 12):
    labels = np.where(labels == i, i - 2, labels)

for i in range(17, 26):
    labels = np.where(labels == i, i - 5, labels)

labels = np.where(labels == 27, 21, labels)


batch_size = 64
lr = 0.15
num_epochs = 200

model = MLPC(hidden_layer_sizes=(400, 200, 50), batch_size=batch_size, max_iter=num_epochs)
model.fit(data, labels)
score = model.score(data, labels)

print('\n\n')
print(f'{score=}')
print('\n\n')
input()

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = hd(maxHands=1)
while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    # frame = frame[:, (w-h)//2:w-(w-h)//2, :]
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame)
    letter = "nothing"
    if lmlist:
        # lmlist = np.array(lmlist, dtype=float)

        lmlist = preprocess(np.array(lmlist, dtype=float))
        if lmlist.shape != (21, 3):
            continue
        lmlist = lmlist.reshape(1, 21*3)
        # lmlist = lmlist[:, :2] # no z
        # lmlist = torch.from_numpy(lmlist.astype(np.float32))
        # zs = model(lmlist[None, :, :])
        
        # pred = zs.max(1, keepdim=True)[1] 
        pred = model.predict(lmlist)[0]
        # pred = max(zs)
        print(pred)
        letter = newTranslate[pred]
    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.putText(frame, letter, (h-20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break