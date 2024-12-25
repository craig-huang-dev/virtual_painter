import time
import cv2
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence,
                                        self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(RGB_img)
        return img

    def drawLandmarks(self, img, hand_landmarks):
        if self.result.multi_hand_landmarks:
            for hand_landmarks in self.result.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

    def findPosition(self, img, handNo=0, draw=False):
        landmarkList = []
        # Print each landmark
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                landmarkList.append([id, cx, cy])
                # Purple circle on index finger
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return landmarkList


def main():

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    prev_time = 0
    curr_time = 0
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)
        if len(landmarkList) != 0:
            print(landmarkList[8])

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        # Display FPS
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("capture image", img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()
