import cv2
import numpy as np
import tensorflow as tf

# ==== CONFIG ====
MODEL_PATH = "sign_model.h5"
CLASS_NAMES_PATH = "class_names.txt"
IMG_SIZE = (64, 64)   # Must match training size
# =================

def load_class_names(path):
    with open(path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def main():
    # 1. Load model and class names
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)
    print("Loaded classes:", class_names)

    # 2. Open webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip horizontally (mirror view)
        frame = cv2.flip(frame, 1)

        # Define a region of interest (ROI) for the hand (e.g., right side)
        h, w, _ = frame.shape
        x1, y1 = int(w * 0.6), int(h * 0.1)
        x2, y2 = int(w * 0.95), int(h * 0.75)

        # Draw rectangle where the hand should be placed
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        roi = frame[y1:y2, x1:x2]

        # Only predict if ROI has valid size
        if roi.size != 0:
            # 3. Preprocess ROI for model
            img = cv2.resize(roi, IMG_SIZE)
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)  # (1, H, W, 3)

            # 4. Predict
            preds = model.predict(img, verbose=0)
            class_idx = np.argmax(preds[0])
            confidence = float(np.max(preds[0]))
            label = class_names[class_idx]

            # 5. Display prediction on frame
            text = f"{label} ({confidence*100:.1f}%)"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show result
        cv2.imshow("Real-time Sign Language Recognition", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
