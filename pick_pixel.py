import cv2
import os

# Put your Anchor Frame path here (e.g., frame 351)
IMAGE_PATH = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/splat_data/images/frame_122.jpg"

def click_event(event, x, y, flags, params):
    # Check for left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"\n✅ Perfect! Update your script to use:\nCROSSHAIR_PIXEL = ({x}, {y})\n")
        
        # Draw a red dot where you clicked just to confirm
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Click a distinct feature (Press any key to close)', img)

# Load image
img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"Error: Could not load image at {IMAGE_PATH}")
else:
    print("Window opened. Click on a distinct feature (like a crack or corner) on the structure.")
    cv2.imshow('Click a distinct feature (Press any key to close)', img)
    
    # Set mouse handler
    cv2.setMouseCallback('Click a distinct feature (Press any key to close)', click_event)
    
    # Wait for a key press to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()