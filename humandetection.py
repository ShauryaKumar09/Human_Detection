import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import winsound
import time

last_alert = 0
COOLDOWN = 1000000000 # seconds


def play_alert():
    winsound.PlaySound("alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)


# Path to your model (update this if needed)
MODEL_PATH = r"Human detection/efficientdet_lite0.tflite"


SOURCE = "rtsp://Shauryacam:shaurya@192.168.6.153:554/live/ch1"

cap = cv2.VideoCapture(SOURCE, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # less lag (quality unchanged)

print("Stream size:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Load MediaPipe Object Detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.3,
    max_results=5
)
detector = vision.ObjectDetector.create_from_options(options)

WIN = "Person Detection (MediaPipe)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1400, 800)  # adjust to taste



while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    frame = cv2.resize(frame, (int(frame.shape[1] * 1.1), frame.shape[0]))

    # Check if a person is detected
    person_found = False
    if result.detections:
        for det in result.detections:
            cat = det.categories[0]
            label, score = cat.category_name, cat.score


            if label.lower() == "person" and score >= 0.25:
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
        now = time.time()
        if now - last_alert > COOLDOWN:
            last_alert = now
            play_alert()

        
        cv2.putText(frame, "PERSON DETECTED",
                    (200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No person detected",
                    (180, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow(WIN, frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
