import cv2 
import mediapipe as mp

cap = cv2.VideoCapture(0)

# ? cv2 > 
# ?      .VideoCapture() Function
# ?      .read() Function
# ?      .cvtcolor() Function
# ?      .imshow() Function 
# ?      .waitKey() Function

# ? mp > solutions > 
# ?                  .hands > Hands() Function
# ?                  .drawing_utils > .draw_landmarks() Function 

mpHands = mp.solutions.hands  
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils 

pTime = 0 
cTime = 0 

while True:
    # ! access camera
    success, img = cap.read()

    #-------------------------------------------------------- 
    # ! make the hand
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    #--------------------------------------------------------

    # ! show the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)