import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import cv2

MODEL_PATH = "cnn.h5"
LABELS_PATH = "labels.json"
IMG_SIZE = 128

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# Load labels
with open(LABELS_PATH, "r") as f:
    reverse_label_map = json.load(f)
# Convert keys to int if needed
reverse_label_map = {int(k): v for k, v in reverse_label_map.items()}
print("Labels loaded:", reverse_label_map)

# Function to predict on an image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    pred_label = reverse_label_map[pred_idx]
    confidence = preds[0][pred_idx]

    print(f"Predicted: {pred_label} (Confidence: {confidence:.2f})")
    return pred_label, confidence

# real-time webcam prediction
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img / 255.0, axis=0)

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    pred_label = reverse_label_map[pred_idx]
    confidence = preds[0][pred_idx]

    cv2.putText(frame, f"{pred_label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
