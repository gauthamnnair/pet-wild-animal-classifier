# pet-wild-animal-classifier

This is a Flask-based API that utilizes the YOLOv8 model to detect animals in images and videos. The API also provides safety tips for detected animals based on a CSV dataset.

## Features

- Detects animals and people in images and videos.
- Categorizes detected animals as `pet`, `wild`, `farm`, or `person`.
- Provides safety tips and descriptions for detected animals.
- Processes both images and videos, returning annotated results.
- Uses OpenCV for video processing and bounding box visualization.

## Technologies Used

- Python
- Flask
- OpenCV
- YOLOv8 (Ultralytics)
- Pandas
- NumPy
- Base64 encoding for image and video transfer

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Running the Application
```bash
python app.py
```
