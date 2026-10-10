# American-Sign-Language-ASL-Recognition
A deep learning project to classify American Sign Language (ASL) hand signs from images using PyTorch.

The goal of this project is to recognize different American Sign Language (ASL) alphabet hand signs. I trained a Convolutional Neural Network (CNN) in a Jupyter Notebook using PyTorch, and then loaded the trained model into a Python script (main.py) to test it in real time using my webcam.

# Dataset
- Classes: 29 different classes (A–Z, del, nothing, and space). 
- Images: Exactly 3,000 images per class (87,000 images in total).
- Image Size: The original images were 200x200 pixels.Pre-processing: Because loading all those high-resolution images was too heavy for my computer, I resized every image down to 64x64 pixels and normalized pixel values to make training manageable.

# Model Architecture
I built a custom Convolutional Neural Network (CNN) with PyTorch: 
- Convolutional Layers: 3 Conv2D layers with ReLU activations and MaxPooling to detect features.
- Dropout: Set to 0.5 before the fully connected layers to help prevent overfitting.
- Fully Connected Layers: A simple feedforward network at the end that outputs the final 29 classes

# Training Details & Challenges
- Batch Size: 64
- Loss Function: nn.CrossEntropyLoss
- Optimizer Troubles: At first, I tried using standard SGD with a small learning rate (0.001) and no momentum, but the model didn't learn at all (accuracy was stuck around ~3.3%, which is basically just random guessing). I switched to Adam (lr=0.001), and it started learning right away.
-   Final Results: After training for 6 epochs, the model achieved 98.3% accuracy and a loss of 0.05 on the test dataset.

  
  To test how well the model works in real life, I used main.py (provided by my mentor).

  <img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/97a5c70e-f887-4bd8-9a1b-d7ad1c232dbf" />

