import cv2 
import mediapipe as mp
import time

# This line opens the webcam
# 0 means default camera — if it doesn't work, try 1 or 2 depending on your setup
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands  # Accessing the hand tracking module from mediapipe
hands = mpHands.Hands()       # Creating a hands detector object with default settings
mpDraw = mp.solutions.drawing_utils # Load the drawing utilities from Mediapipe to visualize landmarks on the image

pTime = 0 # previous time
cTime = 0 # current

# This is an infinite loop — it keeps running until you forcefully stop it (or press 'q' in our next version)
while True:
    # Reads a frame from the webcam (like taking a picture)
    # success = a boolean True/False depending on whether the frame was read correctly
    # img = the actual image/frame from the webcam  a NumPy array that represents the pixels
    success, img = cap.read() 

    # OpenCV uses BGR (Blue-Green-Red) by default but Mediapipe expects RGB input.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting BGR (default from OpenCV) to RGB for mediapipe processing
    
    results = hands.process(imgRGB)                # Processing the RGB image to detect hands and their landmarks
    # Returns a results object containing various information, especially 
    # 1. results.multi_hand_landmarks — a list of landmarks for each detected hand.
    # 2. results.multi_handedness — info about left/right hand, if available.
    
    # Printing detected hand landmarks (None if no hands are found)
    #* print(results.multi_hand_landmarks)           
    
    # Inside the loop, check if any hand landmarks were detected
    if results.multi_hand_landmarks:
        # Loop through each detected hand
        for handLms in results.multi_hand_landmarks:
            # every dots has some coordinates and its index and we'll print it with this loop
            for id, landmark in enumerate(handLms.landmark): 
                # this gives x,y,z but they are in decimals and too detail 
                #* print(id, landmark) 

                #to counter the too detailed part we'll make this height, width, channels

                # Get the height, width, and channels of the current frame/image (img) from the webcam
                # img.shape gives height: number of rows (pixels), width: number of columns (pixels), channels: color channels (RGB)
                height, width, channels = img.shape 
                
                # Calculate the x and y coordinates of the landmark in pixel values
                # The landmark coordinates are between 0 and 1, so we multiply by the image width and height
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                
                # Print the id of the landmark and its corresponding x, y pixel coordinates (cx, cy)
                print(id, cx, cy)

                # we can even put specific id and make them bigger too
                if id == 0: #make the landmark with id 0 bigger than the rest
                    cv2.circle(img, (cx,cy), 25, (225, 0, 225), cv2.FILLED)
                #! Thumb Tip - ID 4
                #! Index Finger Tip - ID 8
                #! Middle Finger Tip - ID 12
                #! Ring Finger Tip - ID 16
                #! Little Finger Tip - ID 20

            # Draw the hand landmarks on the image (with default style)
            mpDraw.draw_landmarks(
            img,                     # <- The image you want to draw on 
            # meaing you can totally draw the hand landmarks on a blank image, 
            # You’ll still need to capture a frame (img) and process it normally — but instead of drawing on img, you draw on your own canvas.
            
            handLms,                 # <- The landmark data for one hand
            mpHands.HAND_CONNECTIONS,  # Connect the landmarks to form hand structure
            mpDraw.DrawingSpec(color=(255, 0, 255), thickness=3, circle_radius=5),  # Pink dots
            mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)  # Green lines
        )

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {round(fps, 2)}", (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 225, 0), 3, cv2.LINE_AA, False)

    cv2.imshow("Image", img) # Shows the captured image in a window titled "Image"
    
    # Waits for 1 millisecond between frames (helps video flow smoothly)
    # Without this, the window won’t update fast enough or might crash
    #* cv2.waitKey(1)
    
    # This checks if you pressed the 'q' key
    # cv2.waitKey(1) waits 1 ms for a keypress and returns a number (ASCII code)
    # ord('q') is the ASCII value of the letter 'q'
    # The & 0xFF part makes sure it works correctly across different operating systems
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break # If 'q' is pressed, break out of the loop and stop the webcam

# Releases the webcam — closes it properly so other apps can use it
#* cap.release()
# Destroys the OpenCV window we created (otherwise it'll stay open or freeze)
#* cv2.destroyAllWindows()