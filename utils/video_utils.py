import cv2
from tqdm import tqdm


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        print("can't read this video")
        exit(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    process_save = tqdm(output_video_frames, colour= 'cyan', desc="save video")
    for frame in process_save:
        out.write(frame)
        process_save.update(1)
    out.release()


