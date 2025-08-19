# CSC480_Final

# Real-Time Fatigue and Emotion Detection System

A real-time facial analysis system built with Python, OpenCV, and TensorFlow. This project detects user fatigue through blink/yawn analysis and recognizes 4 primary emotions using both a Convolutional Neural Network and a geometric, rule-based approach.

## Features

-   **Fatigue Detection**: Monitors Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) in real-time.
-   **CNN Emotion Recognition**: A fine-tuned MobileNetV2 model classifies 4 emotions (Happy, Sad, Angry, Surprise).
-   **Geometric Emotion Analysis**: A rule-based engine that uses facial landmarks to detect basic expressions like smiles.
-   **Comparative Display**: Shows the results from both the CNN and geometric methods side-by-side for analysis.

## Tech Stack

-   Python 3.11
-   TensorFlow 2.16
-   OpenCV
-   MediaPipe
-   NumPy, SciPy

## Setup and Installation

Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    *On Windows, use `venv\Scripts\activate`*

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    

## How to Run

There are two main scripts in this project.

1.  **To run the live demo:**
    *Make sure your webcam is connected.*
    ```bash
    python app.py
    ```
    *Press 'q' to quit the application.*

2.  **To re-train the model:**
    *The FER-2013 dataset (organized into folders) should be placed in the `data/archive` directory.*
    ```bash
    python train_v2.py
    ```

## Project Structure

    .
    ├── data/                 # Contains the dataset (not committed)
    ├── emotion_model_v3_finetuned.keras  # The trained model
    ├── app.py                # Main script to run the live demo
    ├── train_v2.py           # Script to train the CNN model
    ├── prepare_data.py       # Helper script for data loading
    ├── requirements.txt      # Project dependencies
    └── README.md             # This file