# Real-Time Emotion Recognition System

A deep learning-based real-time facial emotion recognition system built using Python, OpenCV, and PyTorch.

The system detects faces through a webcam feed and classifies emotions into multiple categories in real time.

---

## Features

- Real-time webcam emotion detection
- Face detection using OpenCV
- Deep learning-based emotion classification
- Supports multiple emotion categories:
  - Angry
  - Happy
  - Sad
  - Fear
  - Surprise
  - Neutral
  - Disgust
- Prediction smoothing for stable outputs
- Model evaluation using confusion matrix and classification report

---

## Tech Stack

### Languages
- Python

### Libraries & Frameworks
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow (PIL)

---

## Project Structure

```bash
Real-Time-Emotion-Recognition-System/
│
├── models/
├── train/
├── test/
├── outputs/
├── realtime_detection.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/ritWiz564/Real-Time-Emotion-Recognition-System.git
cd Real-Time-Emotion-Recognition-System
```

### Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate virtual environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python realtime_detection.py
```

---

## Model Training

```bash
python train.py
```

---

## Evaluation

```bash
python evaluate.py
```

---

## Applications

- Human-computer interaction
- Smart surveillance systems
- Mental health monitoring
- Engagement analysis
- AI-based emotion analytics

---

## Future Improvements

- Improve model accuracy
- Deploy as web application
- Add voice emotion recognition
- Mobile support
- Multi-face tracking

---

## Author

Ritvika Surana

---

## License

This project is licensed under the MIT License.