import cv2
import numpy as np

# Define the lower and upper boundaries of the colors in the HSV color space
colors = {
    "red": ([0, 100, 100], [10, 255, 255]),
    "orange": ([10, 100, 100], [20, 255, 255]),
    "yellow": ([20, 100, 100], [30, 255, 255]),
    "green": ([30, 100, 100], [60, 255, 255]),
    "blue": ([90, 100, 100], [120, 255, 255]),
    "purple": ([120, 100, 100], [150, 255, 255]),
    "pink": ([150, 100, 100], [170, 255, 255]),
}

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Loop over the colors and detect them in the frame
    for color_name, (lower, upper) in colors.items():
        # Create a mask for the color
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Find the contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over the contours and draw a bounding box around each one
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w*h > 500:
                cv2.putText(frame, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Color Detection", frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
