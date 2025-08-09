import cv2
from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
from argparse import ArgumentParser
import numpy as np
import os
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator

def get_args():
    parser = ArgumentParser(description="football analysis")
    parser.add_argument("--input_video", "-i", type=str, default="./video/docmun.mp4", help="path to video input")
    parser.add_argument("--output_video", "-o", type=str, default="./output_video/output_video.avi", help= "path to video output")
    parser.add_argument("--path_model", "-m", type=str, default="./model/best.pt", help="path to model detect")
    parser.add_argument("--read_stubs", action="store_true", help="Read stub")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    #read video
    video_frames = read_video(args.input_video)

    #initialize tracker
    tracker = Tracker(args.path_model)

    # process_stubs name
    filename = os.path.basename(args.input_video)
    name_only = os.path.splitext(filename)[0]
    stub_detect = f"./stubs/" + "detect_"+ str(name_only) + ".pkl"
    stubs_cam = f"./stubs/" + "camera_"+ str(name_only) + ".pkl"

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=args.read_stubs, stub_path=stub_detect)

    #get object position
    tracker.add_position_to_tracks(tracks)
    #camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stubs=args.read_stubs , stub_path= stubs_cam)

    camera_movement_estimator.add_adjust_position_to_tracks(tracks, camera_movement_per_frame)
    #imterpolate missing value ball
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    #Assige team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    #Assign ball aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)
    # team_ball_control = np.array(team_ball_control)


    #draw output
    ##drwa object track
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    #Save video
    save_video(output_video_frames, args.output_video)
    print(f"✅ Output video has been saved to: {args.output_video}")


if __name__ == '__main__':

    main()
