import cv2
import ht_Module as htm 
import numpy as np
import math

# folderpath = "Header"
# mylist = os.listdir(folderpath)
# print(mylist)
# overlaylist = []
# for impath in mylist:
#     image = cv2.imread(f'{folderpath}/{impath}')
#     overlaylist.append(image)
# print(len(overlaylist))
# header = overlaylist[0]

drawcolor = (255, 0, 255)  # Default color for drawing (purple)
brushthickness = 15  # Brush thickness for drawing

xp, yp = 0, 0  # Previous position for continuous drawing

# Create a blank canvas to draw on (720 height, 1280 width, 3 color channels)
imgcanvas = np.zeros((720, 1280, 3), np.uint8)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1288)  # Set the webcam width to 1280
cap.set(4, 720)   # Set the webcam height to 720

# Initialize hand detector
detector = htm.handDetection()

ptime = 0  # For FPS tracking

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Flip the image for natural drawing movement

    # Detect hands and get landmarks
    img = detector.findHands(img)
    lmlist, _ = detector.findPositions(img, draw=False)

    if len(lmlist) != 0:
        # Get index and middle finger tips
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
        # print(lmlist)
        fingers = detector.fingerup()

        # Selection mode: Two fingers up
        if fingers[1] and fingers[2]:
            # print("Selection Mode")
            xp, yp = 0, 0  # Reset drawing start
            if y1 < 125:
                if 250 < x1 < 450:
                    # header = overlaylist[0]
                    drawcolor = (255, 0, 255)  # Purple
                elif 550 < x1 < 750:
                    # header = overlaylist[1]
                    drawcolor = (255, 0, 0)    # Blue
                elif 800 < x1 < 950:
                    # header = overlaylist[2]
                    drawcolor = (0, 255, 0)    # Green
                elif 1050 < x1 < 1250:
                    # header = overlaylist[3]
                    drawcolor = (0, 0, 0)      # Eraser (black)
            # Draw selection rectangle
            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawcolor, cv2.FILLED)

            if lmlist[4][1] > lmlist[3][1]:  # Right hand
                x_thumb, y_thumb = lmlist[4][1:]
                x_index, y_index = lmlist[8][1:]

                length = math.hypot(x_index - x_thumb, y_index - y_thumb)
                brushthickness = int(np.interp(length, [20, 200], [5, 50]))

                # Optional visual feedback
                cv2.line(img, (x_index, y_index), (x_thumb, y_thumb), (255, 255, 255), 2)
                cv2.circle(img, (x_index, y_index), 8, (0, 255, 255), cv2.FILLED)
                cv2.circle(img, (x_thumb, y_thumb), 8, (0, 255, 255), cv2.FILLED)
                cv2.putText(img, f'Thickness: {brushthickness}', (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, drawcolor, 2)

        # Drawing mode: Only index finger up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawcolor, cv2.FILLED)
            # print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Draw on both img and canvas
            cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushthickness)
            cv2.line(imgcanvas, (xp, yp), (x1, y1), drawcolor, brushthickness)

            xp, yp = x1, y1

    # Merge the canvas and the webcam feed
    imggray = cv2.cvtColor(imgcanvas, cv2.COLOR_BGR2GRAY)
    _, imginv = cv2.threshold(imggray, 50, 255, cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imginv)
    img = cv2.bitwise_or(img, imgcanvas)

    # Show the webcam feed with drawing on it
    cv2.imshow("Webcam Feed", img)
    # Optional: show the canvas alone
    # cv2.imshow("Canvas Only", imgcanvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break
