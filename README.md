# Football Analysis Project
=========================

An AI-powered football match analysis system that uses YOLO object detection and computer vision techniques to track players, referees, and the ball in football videos. The system automatically assigns team colors, tracks ball possession, and provides real-time match statistics.

Features
--------

*   **Object Detection & Tracking**: Uses YOLOv5/YOLOv8 to detect and track players, referees, and the ball
    
*   **Team Assignment**: Automatically identifies and assigns team colors using K-means clustering
    
*   **Ball Possession Tracking**: Determines which player has control of the ball at any given time
    
*   **Team Statistics**: Calculates ball control percentages for each team
    
*   **Visual Annotations**: Draws team-colored ellipses around players, triangles for ball possession, and overlays statistics
    
*   **Video Processing**: Processes entire match videos and outputs annotated results
    

Project Structure
-----------------

Plain 

- football_analysis/  
    - ├── main.py                     # Main execution script  
    - ├── yolo_inference.py          # YOLO model inference testing  
    - ├── trackers/                  # Player and object tracking modules  
    - │   ├── __init__.py  
    - │   └── tracker.py            # Main tracking logic with ByteTrack  
    - ├── team_assigner/            # Team identification and color assignment  
    - │   ├── __init__.py  
    - │   └── team_assigner.py     # K-means clustering for team colors  
    - ├── player_ball_assigner/     # Ball possession detection  
    - │   ├── __init__.py  
    - │   └── player_ball_assigner.py  
    - ├── utils/                    # Utility functions  
    - │   ├── __init__.py  
    - │   ├── video_utils.py       # Video reading/writing functions  
    - │   └── bbox_utils.py        # Bounding box utility functions  
    - ├── models/                   # Trained YOLO models  
    - │   └── best.pt              # Custom trained model  
    - ├── input_videos/            # Input video files  
    - ├── output_videos/           # Processed output videos  
    - ├── stubs/                   # Cached tracking data  
    - └── runs/                    # Training runs and results   `

Requirements
------------

### Dependencies

- pip install ultralytics  
- pip install supervision  
- pip install opencv-python  
- pip install pandas  
- pip install numpy  
- pip install scikit-learn  
- pip install pickle5   `

### Hardware Requirements

*   **Recommended**: GPU with CUDA support for faster processing
    
*   **Minimum**: CPU processing (slower but functional)
    
*   **Memory**: 8GB+ RAM recommended for video processing
    

Installation
------------

1.  **Clone the repository**
    

- git clone https://github.com/yourusername/football_analysis.git  
- cd football_analysis   `

2.  **Create virtual environment**
    

- python -m venv cv_env  source cv_env/bin/activate  # On Windows:
- cv_env\Scripts\activate   `

3.  **Install dependencies**
    

- pip install -r requirements.txt   `

4.  **Download or train YOLO model**
    
    *   Place your trained model in models/best.pt
        
    *   Or use the provided model training notebook
        

Usage
-----

### Basic Usage

1.  **Place your video file** in the input\_videos/ directory
    
2.  **Run the analysis**
    

- python main.py   `

3.  **Find the output** in output\_videos/output\_video.mp4
    

Model Training
--------------

The project includes a Jupyter notebook for training custom YOLO models:

1.  **Open the training notebook**
    

- jupyter notebook football_training_yolo_v5.ipynb   `

2.  **Download dataset** (using Roboflow)
    

- from roboflow import Roboflow  rf = Roboflow(api_key="your_api_key")  project = rf.workspace("workspace").project("football-players-detection")  dataset = project.version(1).download("yolov5")   `

3.  **Train the model**
    

- yolo task=detect mode=train model=yolov5x.pt data=path/to/data.yaml epochs=100 imgsz=640   `

How It Works
------------

### 1\. Object Detection

*   Uses YOLO to detect players, referees, and ball in each frame
    
*   Converts goalkeeper detections to player class for consistency
    

### 2\. Object Tracking

*   Implements ByteTrack for consistent object tracking across frames
    
*   Interpolates ball positions for smooth tracking
    

### 3\. Team Assignment

*   Extracts player jersey colors using K-means clustering
    
*   Groups players into two teams based on dominant colors
    
*   Handles edge cases with manual player ID assignments
    

### 4\. Ball Possession

*   Calculates distance between ball and each player
    
*   Assigns ball to closest player within threshold distance
    
*   Tracks possession statistics over time
    

### 5\. Visualization

*   Draws colored ellipses around players (team colors)
    
*   Shows triangles above players with ball possession
    
*   Displays real-time ball control percentages
    

Output Format
-------------

The system generates:

*   **Annotated video** with player tracking and team colors
    
*   **Ball possession indicators** showing current player with ball
    
*   **Statistics overlay** displaying real-time ball control percentages
    
*   **Cached tracking data** for faster re-processing
    

Troubleshooting
---------------

### Common Issues

1.  **Model not found**
    
    *   Ensure models/best.pt exists
        
    *   Check model file path in main.py
        
2.  **Video not loading**
    
    *   Verify video format (MP4 recommended)
        
    *   Check file path in read\_video() call
        
3.  **Memory errors**
    
    *   Reduce batch size in detect\_frames()
        
    *   Process shorter video segments
        
4.  **Poor team detection**
    
    *   Adjust K-means parameters in team\_assigner.py
        
    *   Add manual player ID assignments for problematic cases
        

### Performance Optimization

*   **Use GPU**: Ensure CUDA-compatible PyTorch installation
    
*   **Batch processing**: Adjust batch size based on available memory
    
*   **Stub files**: Enable stub reading for repeated processing
    
*   **Video resolution**: Consider resizing input videos for faster processing
    
