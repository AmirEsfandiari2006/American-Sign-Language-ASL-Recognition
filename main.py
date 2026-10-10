import cv2
import time
import torch
import pyttsx3
from model_def import NeuralNetwork

# Initialize TTS engine once
engine = pyttsx3.init()

# Define classes (must match the training folder sort order)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

# Load the TorchScript model (does NOT require the class definition)
device = torch.device("cpu")
model = NeuralNetwork(num_classes=len(classes))
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()


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

    # Preprocess ROI for PyTorch (64, 64, 3) -> (1, 3, 64, 64)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, IMAGE_SIZE)
    roi_normalized = roi_resized.astype('float32') / 255.0

    # Convert to PyTorch Tensor and change layout from (H, W, C) to (1, C, H, W)
    roi_tensor = torch.tensor(roi_normalized).permute(2, 0, 1).unsqueeze(0).to(device)

    # PyTorch inference
    with torch.no_grad():
        logits = model(roi_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, pred_index = torch.max(probabilities, dim=1)

    pred_index = pred_index.item()
    confidence = confidence.item()
    pred_class = classes[pred_index]

    current_time = time.time()

    # Logic: 5 seconds of 'nothing' to trigger TTS
    if pred_class == 'nothing':
        if nothing_start_time == 0:
            nothing_start_time = current_time
        elif current_time - nothing_start_time >= 5.0 and final_text != "":
            print(f"Speaking: {final_text}")

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
                    final_text += pred_class.lower()

                # Reset to avoid repeated entry of the same char
                current_char = ""
                char_start_time = current_time
        else:
            current_char = ""
            char_start_time = 0

    # Display info on screen
    cv2.putText(frame, f"Pred: {pred_class} ({confidence * 100:.1f}%)", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Text: {final_text}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('ASL Real-time Translation', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()