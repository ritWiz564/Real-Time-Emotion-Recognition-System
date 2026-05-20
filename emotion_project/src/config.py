import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# Training settings
IMG_SIZE = 128
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)