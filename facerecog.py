import cv2

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start capturing video from the default camera (usually 0)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale (required by the face detector)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Wait for the user to press a key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows()