import cv2
import mediapipe as mp

# Initialize the MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Create a canvas to draw on
canvas = np.zeros((720, 1280, 3), np.uint8)

# Loop through each frame from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB format and process it with the hands module
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # If the hands module detected hand landmarks in the frame, draw the index finger tip position on the canvas
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the position of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            print(f"Index finger tip position: ({x}, {y})")

            # Draw a large green circle at the position of the index finger tip on the canvas
            cv2.circle(canvas, (x, y), 50, (0, 255, 0), -1)

    # Display the canvas and the original video frame
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Video", frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources used by the VideoCapture object and the hands module
cap.release()
hands.close()
