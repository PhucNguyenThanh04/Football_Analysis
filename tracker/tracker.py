import torch
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import pandas as pd
import sys
import numpy as np
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox, get_foot_position
from tqdm import tqdm
from scipy.signal import savgol_filter


class Tracker:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)
        self.tracker = sv.ByteTrack()
        self.ball_trajectory = []

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        #interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        process_detect = tqdm(frames, desc="Detect video", colour="cyan")
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1, verbose=False)
            detections += detection_batch
            process_detect.update(batch_size)
        return detections

    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players" :[],
            "referees" : [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = { v:k  for k, v in cls_names.items()}

            #convert to supervision detect format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #convert goalkeeper to player
            for obj_index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[obj_index] = cls_names_inv["player"]

            #tracks object
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])

        x_center, y_center = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes= (int(width), int(0.35*width)),
            angle=0.0,
            startAngle= -45,
            endAngle=235,
            color= color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2-rectangle_height//2) +15
        y2_rect = (y2+rectangle_height//2) +15
        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )
            x1_text = x1_rect +12
            if track_id > 99:
                x1_text -=10
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
                )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_point = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_point],0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_point],0,(0, 0, 0),2)
        return frame

    def draw_line_ball(self, frame, ball_track, color= None):
        for i in range(1, len(ball_track)):
            pt1 = self.ball_trajectory[i - 1]
            pt2 = self.ball_trajectory[i]
            cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlays = frame.copy()
        cv2.rectangle(overlays, (1190, 60), (1852, 148),(255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlays, alpha, frame, 1-alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame ==1].shape[0]
        # team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame ==2].shape[0]
        team_1_num_frames = team_ball_control_till_frame.count(1)
        team_2_num_frames = team_ball_control_till_frame.count(2)
        # team_1 = team_1_num_frames/(team_1_num_frames+ team_2_num_frames)
        # team_2 = team_2_num_frames/(team_1_num_frames+ team_2_num_frames)
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames > 0:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        else:
            team_1 = 0.0
            team_2 = 0.0
        cv2.putText(frame, f"Team 1 control: {int(team_1*100)}%", (1210, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 control: {int(team_2*100)}%", (1210, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame


    def draw_annotations(self, video_frames, tracks, team_ball_control):
        out_put_videoframes = []
        process_draw = tqdm(video_frames, colour='red', desc="visualitation")
        for frame_num, frame in enumerate(process_draw):

            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #draw player
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 0, 255))
            # draw referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            # draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
                ball_center = get_center_of_bbox(ball["bbox"])
                self.ball_trajectory.append(ball_center)
            # frame = self.draw_line_ball(frame,self.ball_trajectory)

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            out_put_videoframes.append(frame)
            process_draw.update(1)
        return out_put_videoframes






