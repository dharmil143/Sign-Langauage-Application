# Sign Language Application

A real-time American Sign Language (ASL) detection application using deep learning and computer vision. The system can recognize 29 different classes including the ASL alphabet (A-Z) and special characters (del, space, nothing).

## Overview

This project implements a Convolutional Neural Network (CNN) to classify American Sign Language gestures in real-time using a webcam. The model is trained on a dataset of ASL alphabet images and achieves over 99% accuracy on the test set.

## Features

- Real-time ASL gesture recognition
- Support for 29 different classes:
  - Complete ASL alphabet (A-Z)
  - Special characters (del, space, nothing)
- High accuracy (99.59% on test set)
- Live webcam input processing

## Prerequisites

- Python 3.x
- OpenCV (cv2)
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dharmil143/Sign-Langauage-Application.git
cd Sign-Langauage-Application
```

2. Install required packages:
```bash
pip install opencv-python tensorflow numpy matplotlib scikit-learn
```

## Project Structure

- `notebook.ipynb`: Jupyter notebook containing model training code
- `code.py`: Real-time webcam detection script
- `Saved_model.h5`: Trained model weights

## Model Architecture

The CNN architecture consists of:
- 3 Convolutional layers (64, 128, 256 filters)
- MaxPooling layers
- Batch Normalization
- Dropout layers (0.2)
- Dense layers (1024 units, 29 output classes)
- Input shape: (32, 32, 3)

## Training Details

- Dataset split: 90% training, 10% testing
- Batch size: 32
- Epochs: 10
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Categorical Crossentropy

Training Results:
- Training accuracy: 99.30%
- Validation accuracy: 99.50%
- Test accuracy: 99.59%

## Usage

1. Train the model (optional, if you want to retrain):
   - Open and run `notebook.ipynb`
   - Update file paths for your dataset
   - The trained model will be saved as 'Saved_model.h5'

2. Run real-time detection:
```python
python code.py
```

3. Hold your hand gesture in front of the webcam to see the predictions.

## Dataset Structure

The dataset should be organized as follows:
```
asl_alphabet_train/
    A/
    B/
    C/
    ...
    Z/
    del/
    nothing/
    space/

asl_alphabet_test/
    A/
    B/
    C/
    ...
    Z/
    del/
    nothing/
    space/
```

## Model Performance

The model shows excellent performance metrics:
- Training accuracy: 99.30%
- Validation accuracy: 99.50%
- Test accuracy: 99.59%
- Test loss: 0.0127

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

- ASL Alphabet dataset contributors
- TensorFlow and OpenCV communities

