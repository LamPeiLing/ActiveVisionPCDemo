import torch
import onnxruntime as ort
import os
import cv2
import time
from torchvision.transforms import ToTensor


class PersonDetector():

  def inference(self, frames):
      
      model_path = os.path.join(os.path.dirname(__file__), "model.onnx")
      sess = ort.InferenceSession(model_path)

      frame_count = 0
      box_player = []
      for frame in frames:

          original_height, original_width = frame.shape[:2]
          # Convert BGR (OpenCV) to RGB format
          im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          # Resize the image to (640, 640)
          im_rgb = cv2.resize(im_rgb, (640, 640))
          # Convert to tensor
          im_data = ToTensor()(im_rgb)[None]

          size = torch.tensor([[640, 640]])

          start_time = time.time()
          output = sess.run(
              output_names=None,
              input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}
          )
          inference_time = time.time() - start_time

          labels, boxes, scores = output

          thrh = 0.6

          scr = scores[0]
          lab = labels[0][scr > thrh]
          box = boxes[0][scr > thrh]

          # Calculate the scaling ratios
          width_ratio = original_width / 640
          height_ratio = original_height / 640

          # Resize the bounding boxes back to the original dimensions
          resized_boxes = []
          for b in box:
                # Map the bounding box back to the original frame size
                resized_box = [
                    int(b[0] * width_ratio),  # x1
                    int(b[1] * height_ratio),  # y1
                    int(b[2] * width_ratio),  # x2
                    int(b[3] * height_ratio)   # y2
                ]
                resized_boxes.append(resized_box)
          box_player.append(resized_boxes)


          frame_count += 1

      if len(box_player) == 0:
        print("No players detected")
      
      return box_player


if __name__ == "__main__":
    inference()