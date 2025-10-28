import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



  
# Path to your model (update this if needed)
MODEL_PATH = r"C:\Shaurya\SKLEARN\Review Machine Learning\Learning Computer Vison\models\efficientdet_lite0.tflite"


SOURCE = 0

# Load MediaPipe Object Detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    max_results=5
)
detector = vision.ObjectDetector.create_from_options(options)

# Connect to camera
cap = cv2.VideoCapture(SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    # Check if a person is detected
    person_found = False
    if result.detections:
        for det in result.detections:
            cat = det.categories[0]
            label, score = cat.category_name, cat.score


            if label.lower() == "person" and score >= 0.50:
                person_found = True
                # Draw bounding box
                box = det.bounding_box
                x, y, w, h = box.origin_x, box.origin_y, box.width, box.height
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {score:.2f}",
                            (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

    # Display alert
    if person_found:
        cv2.putText(frame, "SOMEONE IS IN YOUR ROOM",
                    (400, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 5, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No person detected",
                    (460, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Person Detection (MediaPipe)", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
