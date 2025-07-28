import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape  # Get video dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use the correct MP4 codec
    output_path = output_path.replace(".avi", ".mp4")  # Ensure correct file extension
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
