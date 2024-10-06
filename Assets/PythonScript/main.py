import cv2
import numpy as np
import sys
import os
from court_reference import CourtReference
from bounce_detector import BounceDetector
from ball_detector import BallDetector
from inference_video import PersonDetector
from court_detector import CourtDetector
import argparse
import torch

def read_video(path_video):
    
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break    
    cap.release()
    return frames, fps

def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img

def write_video(frames, homography_matrices, kps_court, ball_track, bounces, boxes, path_output_video, fps, trace=7):
    """ Write .avi file with detected ball tracks
    :params
        frames: list of original video frames
        ball_track: list of ball coordinates
        path_output_video: path to output video
        fps: frames per second
        trace: number of frames with detected trace
    """
    width_minimap = 166
    height_minimap = 350
    
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (width, height))
    
    court_img = get_court_img()
    # court_img, width_minimap, height_minimap = get_court_img()

    for num in range(len(frames)):
        frame = frames[num]
        inv_mat = homography_matrices[num]

        # draw ball track
        for i in range(trace):
          if (num-i > 0):

            if ball_track[num-i][0]:
              x = int(ball_track[num-i][0])
              y = int(ball_track[num-i][1])
              frame = cv2.circle(frame, (x,y), radius=0, color=(0, 0, 255), thickness=10-i)
            else:
              break
                

        # draw court lines
        if kps_court[num] is not None:
          for j in range(0, len(kps_court[num]), 4):
            x1, y1, x2, y2 = kps_court[j][j], kps_court[j][j + 1], kps_court[j][j + 2], kps_court[j][j+ 3]
            frame = cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)

        # draw player boxes
        for b in boxes[num]:
          cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        
        # draw bounce in minimap
        if num-i in bounces and inv_mat is not None:
          ball_point = ball_track[num-i]
          ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
          ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
          court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                                       radius=0, color=(0, 255, 255), thickness=50)

        minimap = court_img.copy()

        coord = ball_track[num]

        # Ball locations in minimap
        if coord is not None:
          p = np.array(coord,dtype='float64')
          ball_pos = np.array([p[0].item(), p[1].item()]).reshape((1, 1, 2))
          transformed = cv2.perspectiveTransform(ball_pos, inv_mat)[0][0].astype('int64')
          cv2.circle(minimap, (transformed[0], transformed[1]), 35, (0,255,255), -1)
        else:
          pass
                
        # concatenate minimap with output video
        minimap = cv2.resize(minimap, (width_minimap, height_minimap))
        frame[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap

        out.write(frame) 
    out.release()   


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_input_video', type=str, help='path to input video')
    parser.add_argument('--path_output_video', type=str, help='path to output video')
    args = parser.parse_args()

    # input_video = sys.argv[1]
    # output_video = sys.argv[2]
    input_video = args.path_input_video
    output_video = args.path_output_video
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps'
    # frames, fps = read_video(args.path_input_video)  
    frames, fps = read_video(input_video)


    # person detection
    print('player detection')
    person_detector = PersonDetector()
    boxes = person_detector.inference(frames)
    
    print('court detection')
    court_detector = CourtDetector()
    homography_matrices, kps_court = court_detector.infer_model(frames)  

    print('ball detection')
    ball_detector = BallDetector(os.path.join(os.path.dirname(__file__), "model_best.pt"), device)
    ball_track = ball_detector.infer_model(frames)

    # bounce detection
    bounce_detector = BounceDetector(os.path.join(os.path.dirname(__file__), "ctb_regr_bounce.cbm"))
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    # write_video(frames, homography_matrices, kps_court, ball_track, bounces, boxes, args.path_output_video, fps)
    write_video(frames, homography_matrices, kps_court, ball_track, bounces, boxes, output_video, fps)




