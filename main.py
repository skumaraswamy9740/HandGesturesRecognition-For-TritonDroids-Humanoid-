import numpy as np
import cv2

# Constants
FRAME_HEIGHT = 200
FRAME_WIDTH = 300
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18

# Region of Interest (ROI) coordinates
REGION_TOP = 0
REGION_BOTTOM = int(2 * FRAME_HEIGHT / 3)
REGION_LEFT = int(FRAME_WIDTH / 2)
REGION_RIGHT = FRAME_WIDTH

# Global variables
background = None
frames_elapsed = 0
hand = None


class HandData:
    """Stores hand properties and performs gesture analysis."""
    
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0
        self.isInFrame = False
        self.isWaving = False
        self.fingers = None
        self.gestureList = []

    def update_position(self, top, bottom, left, right):
        """Updates hand position coordinates."""
        self.top, self.bottom, self.left, self.right = top, bottom, left, right

    def detect_waving(self, centerX):
        """Detects waving motion based on hand movement."""
        self.prevCenterX, self.centerX = self.centerX, centerX
        self.isWaving = abs(self.centerX - self.prevCenterX) > 3


def write_on_image(frame, hand):
    """Displays gesture-related text and highlights ROI."""
    
    if frames_elapsed < CALIBRATION_TIME:
        text = "Calibrating..."
    elif hand is None or not hand.isInFrame:
        text = "No hand detected"
    else:
        text = "Waving" if hand.isWaving else {
            0: "Rock",
            1: "Pointing",
            2: "Scissors"
        }.get(hand.fingers, "Unknown Gesture")

    # Display text on screen
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Highlight ROI
    cv2.rectangle(frame, (REGION_LEFT, REGION_TOP), (REGION_RIGHT, REGION_BOTTOM), (255, 255, 255), 2)


def get_region(frame):
    """Extracts and preprocesses the ROI."""
    region = frame[REGION_TOP:REGION_BOTTOM, REGION_LEFT:REGION_RIGHT]
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(region, (5, 5), 0)


def get_average(region):
    """Computes a weighted background average for segmentation."""
    global background
    if background is None:
        background = region.astype("float")
    else:
        cv2.accumulateWeighted(region, background, BG_WEIGHT)


def segment(region):
    """Segments the hand from the background."""
    global hand
    diff = cv2.absdiff(background.astype(np.uint8), region)
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    
    contours, _ = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        if hand:
            hand.isInFrame = False
        return None
    
    if hand:
        hand.isInFrame = True
    return thresholded_region, max(contours, key=cv2.contourArea)


def get_hand_data(thresholded_image, segmented_image):
    """Extracts hand extremities and determines gesture."""
    global hand

    convexHull = cv2.convexHull(segmented_image)
    
    # Extract hand extremities
    top = tuple(convexHull[convexHull[:, :, 1].argmin()][0])
    bottom = tuple(convexHull[convexHull[:, :, 1].argmax()][0])
    left = tuple(convexHull[convexHull[:, :, 0].argmin()][0])
    right = tuple(convexHull[convexHull[:, :, 0].argmax()][0])

    centerX = (left[0] + right[0]) // 2

    # Initialize or update hand data
    if hand is None:
        hand = HandData(top, bottom, left, right, centerX)
    else:
        hand.update_position(top, bottom, left, right)

    if frames_elapsed % 6 == 0:
        hand.detect_waving(centerX)

    # Update gesture recognition every 12 frames
    hand.gestureList.append(count_fingers(thresholded_image))
    if frames_elapsed % 12 == 0:
        hand.fingers = most_frequent(hand.gestureList)
        hand.gestureList.clear()


def count_fingers(thresholded_image):
    """Counts fingers based on contour intersections."""
    line_height = int(hand.top[1] + 0.2 * (hand.bottom[1] - hand.top[1]))

    # Draw horizontal line for finger detection
    line = np.zeros(thresholded_image.shape[:2], dtype=np.uint8)
    cv2.line(line, (thresholded_image.shape[1], line_height), (0, line_height), 255, 1)

    # Detect intersections
    line = cv2.bitwise_and(thresholded_image, thresholded_image, mask=line.astype(np.uint8))
    contours, _ = cv2.findContours(line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fingers = sum(5 < len(curr) < (3 * abs(hand.right[0] - hand.left[0]) / 4) for curr in contours)
    return fingers


def most_frequent(input_list):
    """Finds the most frequently occurring value in a list."""
    return max(set(input_list), key=input_list.count)


def main():
    global frames_elapsed, hand  # Ensure 'hand' is recognized as a global variable
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)

        region = get_region(frame)
        if frames_elapsed < CALIBRATION_TIME:
            get_average(region)
        else:
            segmented = segment(region)
            if segmented:
                thresholded_region, segmented_region = segmented
                cv2.drawContours(region, [segmented_region], -1, (255, 255, 255))
                cv2.imshow("Segmented Image", region)
                
                # Ensure hand data is properly assigned
                get_hand_data(thresholded_region, segmented_region)

        # Ensure 'hand' is passed to the function only if it's initialized
        if hand is None:
            hand = HandData((0, 0), (0, 0), (0, 0), (0, 0), 0)  # Default placeholder values

        write_on_image(frame, hand)  # Now 'hand' will never be undefined
        cv2.imshow("Camera Input", frame)

        frames_elapsed += 1

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
