import cv2
import time
import argparse
from ultralytics import YOLO

# Argument Parsing
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Camera device ID or video file path")
    parser.add_argument("--width", type=int, default=960, help="Video capture width")
    parser.add_argument("--height", type=int, default=540, help="Video capture height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video processing")
    parser.add_argument("--skip_frame", type=int, default=0, help="Number of frames to skip")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLOv8 model weights")
    parser.add_argument("--score_th", type=float, default=0.7, help="Detection confidence threshold")
    return parser.parse_args()

args = get_args()

# Class mapping for Naruto hand seals
class_mapping = {
    0: "Bird",
    1: "Boar",
    2: "Dog",
    3: "Dragon",
    4: "Hare",
    5: "Horse",
    6: "Monkey",
    7: "Ox",
    8: "Ram",
    9: "Rat",
    10: "Snake",
    11: "Tiger"
}

# Load YOLO model
model = YOLO(args.model)

# Initialize frame count and FPS calculation
frame_count = 0
predictions = []

def techniques(frame, predictions, scale):
    # Get the height and width of the frame
    f_h, f_w = frame.shape[0:2]
    
    # Set the starting position for the first image at the bottom-left corner
    prev_w = 10  # Add a margin from the left
    prev_h = f_h - 100  # Position images close to the bottom with some margin
    
    frame = frame.copy()
    
    for pred in predictions:
        # Read the image corresponding to the prediction
        pic = cv2.imread("{}.png".format(pred))
        
        # Resize the image based on the given scale
        p_w = int(pic.shape[1] * scale / 100)
        p_h = int(pic.shape[0] * scale / 100)
        pic = cv2.resize(pic, (p_w, p_h))
        
        # Calculate the region for placing the image
        top = prev_h
        bottom = prev_h + p_h
        left = prev_w
        right = prev_w + p_w
        
        # Ensure the image doesn't go beyond the frame's width
        if right > f_w:
            break  # Exit if there's no more space for images
        
        # Add the image on top of the frame at the calculated position
        added_image = cv2.addWeighted(frame[top:bottom, left:right, :], 1, pic[0:p_h, 0:p_w, :], 1, 0)
        frame[top:bottom, left:right, :] = added_image
        
        # Update the position for the next image (move horizontally)
        prev_w += p_w + 10  # Add a little margin between images
    
    return frame

# Main Processing Loop
cap = cv2.VideoCapture(args.device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_FPS, args.fps)

pause = 0

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % (args.skip_frame + 1) != 0:
        continue

    # Perform inference with YOLO
    results = model(frame)

    # Process results (detections)
    class_ids = []
    for result in results:  # `results` is a list
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if conf < args.score_th:
                continue
            class_id = int(cls)
            class_name = class_mapping.get(class_id, f"Class {class_id}")
            class_ids.append(class_name)

    # Update predictions based on detections
    if len(class_ids) > 0:
        predictions.extend(class_ids)
        predictions = list(set(predictions))

    # Add black horizontal bars and display FPS
    f_h, f_w = frame.shape[0:2]
    bar_thickness = 50  # Thickness of the black bar

    # Add black bar on top
    frame[:bar_thickness, :] = (0, 0, 0)  # Set top bar to black

    # Add black bar on bottom
    frame[f_h-bar_thickness:, :] = (0, 0, 0)  # Set bottom bar to black

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    # Display predictions in the frame
    if len(predictions) != 0:
        frame = techniques(frame, predictions, 20)

    # Display the processed frame
    cv2.imshow("Naruto Hand Seals Detection", frame)

    # Exit condition: press 'Esc' to quit
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the Escape key
       break


cap.release()
cv2.destroyAllWindows()