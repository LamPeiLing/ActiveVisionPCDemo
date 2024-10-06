from model import BallTrackerNet
import torch
import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
import cv2
from itertools import groupby

class BallDetector:
    def __init__(self, path_model=None, device='mps'):
        self.model = BallTrackerNet()
        self.path = path_model
        self.device = device
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
        self.width = 640
        self.height = 360


    def infer_model(self, frames):
      """ Run pretrained model on a consecutive list of frames
      :params
          frames: list of consecutive video frames
          model: pretrained model
      :return
          ball_track: list of detected ball points
          dists: list of euclidean distances between two neighbouring ball points
      """
      height = 360
      width = 640
      dists = [-1] * 2
      ball_track = [(None, None)] * 2
    
      original_height, original_width = frames[0].shape[:2]
    
      for num in tqdm(range(2, len(frames))):
          img = cv2.resize(frames[num], (width, height))
          img_prev = cv2.resize(frames[num-1], (width, height))
          img_preprev = cv2.resize(frames[num-2], (width, height))
          imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
          imgs = imgs.astype(np.float32) / 255.0
          imgs = np.rollaxis(imgs, 2, 0)
          inp = np.expand_dims(imgs, axis=0)
    
          out = self.model(torch.from_numpy(inp).float().to(self.device))
          output = out.argmax(dim=1).detach().cpu().numpy()
    
    
          x_pred, y_pred = self.postprocess(output)
    
    
          if x_pred is not None and y_pred is not None:
              x_pred = int(x_pred * original_width / width)
              y_pred = int(y_pred * original_height / height)
    
          ball_track.append((x_pred, y_pred))
    
          if ball_track[-1][0] and ball_track[-2][0]:
              dist = distance.euclidean(ball_track[-1], ball_track[-2])
          else:
              dist = -1
          dists.append(dist)
    
      ball_track = self.remove_outliers(ball_track, dists)
      subtracks = self.split_track(ball_track)
      for r in subtracks:
          ball_subtrack = ball_track[r[0]:r[1]]
          ball_subtrack = self.interpolation(ball_subtrack)
          ball_track[r[0]:r[1]] = ball_subtrack
    
      return ball_track


    def postprocess(self, feature_map):
      feature_map *= 255
      feature_map = feature_map.reshape((360, 640))
      feature_map = feature_map.astype(np.uint8)
      ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
      circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
      x, y = None, None
      if circles is not None:
          if len(circles) == 1:
              x = circles[0][0][0]
              y = circles[0][0][1]
      return x, y
    
    def remove_outliers(self, ball_track, dists, max_dist = 100):
      """ Remove outliers from model prediction    
      :params
          ball_track: list of detected ball points
          dists: list of euclidean distances between two neighbouring ball points
          max_dist: maximum distance between two neighbouring ball points
      :return
          ball_track: list of ball points
      """
      outliers = list(np.where(np.array(dists) > max_dist)[0])
      for i in outliers:
          if (dists[i+1] > max_dist) | (dists[i+1] == -1):       
              ball_track[i] = (None, None)
              outliers.remove(i)
          elif dists[i-1] == -1:
              ball_track[i-1] = (None, None)
      return ball_track  

    def split_track(self, ball_track, max_gap=4, max_dist_gap=80, min_track=5):
      """ Split ball track into several subtracks in each of which we will perform
      ball interpolation.    
      :params
          ball_track: list of detected ball points
          max_gap: maximun number of coherent None values for interpolation  
          max_dist_gap: maximum distance at which neighboring points remain in one subtrack
          min_track: minimum number of frames in each subtrack    
      :return
          result: list of subtrack indexes    
      """
      list_det = [0 if x[0] else 1 for x in ball_track]
      groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

      cursor = 0
      min_value = 0
      result = []
      for i, (k, l) in enumerate(groups):
          if (k == 1) & (i > 0) & (i < len(groups) - 1):
              dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
              if (l >=max_gap) | (dist/l > max_dist_gap):
                  if cursor - min_value > min_track:
                      result.append([min_value, cursor])
                      min_value = cursor + l - 1        
          cursor += l
      if len(list_det) - min_value > min_track: 
          result.append([min_value, len(list_det)]) 
      return result    

    def interpolation(self,coords):
      """ Run ball interpolation in one subtrack    
      :params
          coords: list of ball coordinates of one subtrack    
      :return
          track: list of interpolated ball coordinates of one subtrack
      """
      def nan_helper(y):
          return np.isnan(y), lambda z: z.nonzero()[0]

      x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
      y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

      nons, yy = nan_helper(x)
      x[nons]= np.interp(yy(nons), yy(~nons), x[~nons])
      nans, xx = nan_helper(y)
      y[nans]= np.interp(xx(nans), xx(~nans), y[~nans])

      track = [*zip(x,y)]
      return track
      
