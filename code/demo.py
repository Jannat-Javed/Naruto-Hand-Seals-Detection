import cv2
import time
import argparse
import copy
from ultralytics import YOLO


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


def draw_detections(frame, results, class_mapping, score_th):
    for result in results:  # results is a list
        # Loop through detected boxes
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if conf < score_th:
                continue
            
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            
            # Get class name using the mapping
            class_id = int(cls)
            class_name = class_mapping.get(class_id, f"Class {class_id}")
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display class name and confidence
            text = f"{class_name} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame


def main():
    args = get_args()

    # Class mapping for Naruto hand seals
    class_mapping = {
        0: "Bird-",
        1: "Boar-",
        2: "Dog-",
        3: "Dragon-",
        4: "Hare-",
        5: "Horse-",
        6: "Monkey-",
        7: "Ox-",
        8: "Ram-",
        9: "Rat-",
        10: "Snake-",
        11: "Tiger-"
    }

    # Load YOLO model
    model = YOLO(args.model)

    # Initialize video capture
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    frame_count = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame_count += 1
        if frame_count % (args.skip_frame + 1) != 0:
            continue

        # Perform inference
        results = model(frame)

        # Draw detections
        debug_frame = copy.deepcopy(frame)
        debug_frame = draw_detections(debug_frame, results, class_mapping, args.score_th)

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time
        cv2.putText(debug_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display output
        cv2.imshow("Naruto Hand Seals Detection", debug_frame)

        # Exit condition: press 'Esc' to quit
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the Escape key
          break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 