from deepface import DeepFace
import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Use DeepFace for emotion recognition
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    # Get the dominant emotion
    emotion = result[0]['dominant_emotion']

    # Display the emotion label on the frame
    cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame with the detected emotion
    cv2.imshow('Real-Time Emotion Detection - DeepFace', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
