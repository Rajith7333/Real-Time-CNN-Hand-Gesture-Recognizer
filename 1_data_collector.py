import cv2
import os
import time
import mediapipe as mp

# Labels for which we want to collect images
LABELS = [0, 1, 2, 3, 4, 5]
# Change the integer to increase the number of images of the label folders 
IMAGES_PER_LABEL = 10

# Base folder to save dataset
BASE_PATH = "ds"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Press 'q' to quit at any time.")

for label in LABELS:
    label_path = f"{BASE_PATH}/{label}"
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    
    print(f"\nCollecting images for label {label}. Press 'c' to capture each image.")
    img_count = 0
    
    while img_count < IMAGES_PER_LABEL:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1)

        if key == ord('c'):
            img_name = f"{label_path}/{time.time()}.jpg"
            cv2.imwrite(img_name, frame)
            img_count += 1
            print(f"Saved {img_count}/{IMAGES_PER_LABEL} images for label {label}")

        elif key == ord('q'):
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
print("\nDataset collection complete!")
