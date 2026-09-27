import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

# Load the trained model
model = load_model('best_asl_model.h5')

# Define classes
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

# Initialize webcam and box coordinates
cap = cv2.VideoCapture(0)
x, y, w, h = 300, 100, 300, 300
IMAGE_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.8

# Variables for logic
current_char = ""
char_start_time = 0
nothing_start_time = 0
final_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect and draw bounding box
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = frame[y:y + h, x:x + w]

    # Preprocess ROI for the model
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, IMAGE_SIZE)
    roi_normalized = np.array(roi_resized, dtype='float32') / 255.0
    roi_reshaped = np.expand_dims(roi_normalized, axis=0)

    # Model prediction
    predictions = model.predict(roi_reshaped, verbose=0)
    pred_index = np.argmax(predictions)
    pred_class = classes[pred_index]
    confidence = predictions[0][pred_index]

    current_time = time.time()

    # Logic: 5 seconds of 'nothing' to trigger TTS
    if pred_class == 'nothing':
        if nothing_start_time == 0:
            nothing_start_time = current_time
        elif current_time - nothing_start_time >= 5.0 and final_text != "":
            print(f"Speaking: {final_text}")

            # Convert text to speech
            engine.say(final_text)
            engine.runAndWait()

            # Clear text for the next words
            final_text = ""
            nothing_start_time = 0

        current_char = ""
        char_start_time = 0
    else:
        # Reset 'nothing' timer
        nothing_start_time = 0

        # Logic: 2 seconds of consistent high-confidence char
        if confidence > CONFIDENCE_THRESHOLD:
            if pred_class != current_char:
                current_char = pred_class
                char_start_time = current_time
            elif current_time - char_start_time >= 2.0:
                if pred_class == 'space':
                    final_text += " "
                elif pred_class == 'del':
                    final_text = final_text[:-1]
                else:
                    final_text += pred_class

                # Reset to avoid repeated entry of the same char
                current_char = ""
                char_start_time = current_time
        else:
            current_char = ""
            char_start_time = 0

    # Display info on screen
    cv2.putText(frame, f"Pred: {pred_class} ({confidence * 100:.1f}%)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)
    cv2.putText(frame, f"Text: {final_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('ASL Real-time Translation', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()