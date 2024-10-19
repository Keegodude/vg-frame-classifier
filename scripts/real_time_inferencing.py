import time
import torch
import cv2
from cnn_Model import ScreenshotToneClassifier
import torchvision.transforms as transforms
from PIL import Image
import mss
import pygetwindow as gw
import numpy as np


# Model definition and GPU config
model = ScreenshotToneClassifier()
model.load_state_dict(torch.load('../saved models/best_model.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Helper function to prepare video frame for tone prediction
def preprocess_frame(frame):
    # Convert the OpenCV image (BGR) to a PIL image (RGB) for Torch processing and predicting
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size used in training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transforms
    frame_tensor = transform(frame_pil)
    frame_tensor = frame_tensor.unsqueeze(0)
    return frame_tensor

def predict_tone(frame_tensor):
    # Move the frame to GPU
    frame_tensor = frame_tensor.to(device)

    # Disable gradient computation (speeds up inference)
    with torch.no_grad():
        output = model(frame_tensor)
        _, predicted_label = torch.max(output, 1)  # Get the predicted class

    return predicted_label.item()

# Map numeric labels to their string definition
label_mapping = {
    0: "Action",
    1: "Horror",
    2: "Scenic"
}
# Make 'window_name' into the name of the window to be captured before runtime
window_name = ""
# Find the game window by title and fetch its boundaries
window = gw.getWindowsWithTitle(window_name)[0]
monitor = {
    'top': window.top,
    'left': window.left,
    'width': window.width,
    'height': window.height
}

# Capture the game window using mss
with mss.mss() as sct:
    while True:
        start_time = time.time()
        screenshot = sct.grab(monitor)

        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Call helper function to perform same preprocessing as in training
        processed_frame = preprocess_frame(frame)

        # Call helper function to predict tone using the model
        tone_prediction = predict_tone(processed_frame)
        tone_string = label_mapping[tone_prediction]

        # Overlay the prediction text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Predicted Tone: {tone_string}"
        position = (10, 50)  # X and Y position for the text
        font_scale = 1.0
        font_color = (0, 255, 0)  # Green text for clarity
        line_type = 2

        # Add the text to the frame
        cv2.putText(frame, text, position, font, font_scale, font_color, line_type)

        # Display the frame with the prediction overlay
        cv2.imshow('Tone Prediction Overlay', frame)

        # Set the desired frame rate
        desired_fps = 60
        frame_interval = 1 / desired_fps  # Time between frames

        # Calculate the time taken and adjust to maintain the desired FPS
        elapsed_time = time.time() - start_time
        time_to_wait = frame_interval - elapsed_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
