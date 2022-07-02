from IPython.display import HTML
import PIL.Image
from base64 import b64encode
import seaborn as sns
import ffmpeg
import numpy as np
import os

def show_video(video_path, video_width = "fill"):
  """
  video_path (str): The path to the video
  video_width: Width for the window the video will be shown in 
  """
  video_file = open(video_path, "r+b").read()

  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")

def create_video(frames_patten='Track/%03d.png', video_file = 'movie.mp4', framerate=25):
  """
  frames_patten (str): The patten to use to find the frames. The default patten looks for frames in a folder called Track. The frames shoud be named 001.png, 002.png, ..., 999.png
  video_file (str): The file the video will be saved in 
  framerate (float): The framerate for the video
  """
  if os.path.exists(video_file):
      os.remove(video_file)
  ffmpeg.input(frames_patten, framerate=framerate).output(video_file).run() 

class VisTrack:
    def __init__(self, unique_colors=400):
        """
        unique_colors (int): The number of unique colors (the number of unique colors dos not need to be greater than the max id)
        """
        self._unique_colors = unique_colors
        self._id_dict = {}
        self.p = np.zeros(unique_colors)
        self._colors = (np.array(sns.color_palette("hls", unique_colors))*255).astype(np.uint8)

    def _get_color(self, i):
        return tuple(self._colors[i])

    def _color(self, i):
        if i not in self._id_dict:
            inp = (self.p.max() - self.p ) + 1 
            if any(self.p == 0):
                nzidx = np.where(self.p != 0)[0]
                inp[nzidx] = 0
            soft_inp = inp / inp.sum()

            ic = np.random.choice(np.arange(self._unique_colors, dtype=int), p=soft_inp)
            self._id_dict[i] = ic

            self.p[ic] += 1

        ic = self._id_dict[i]
        return self._get_color(ic)

    def draw_bounding_boxes(self, im: PIL.Image, bboxes: np.ndarray, ids: np.ndarray,
                        scores: np.ndarray) -> PIL.Image:
        """
        im (PIL.Image): The image 
        bboxes (np.ndarray): The bounding boxes. [[x1,y1,x2,y2],...]
        ids (np.ndarray): The id's for the bounding boxes
        scores (np.ndarray): The scores's for the bounding boxes
        """
        im = im.copy()
        draw = PIL.ImageDraw.Draw(im)

        for bbox, id_, score in zip(bboxes, ids, scores):
            color = self._color(id_)
            draw.rectangle((*bbox.astype(np.int64),), outline=color)

            text = f'{id_}: {int(100 * score)}%'
            text_w, text_h = draw.textsize(text)
            draw.rectangle((bbox[0], bbox[1], bbox[0] + text_w, bbox[1] + text_h), fill=color, outline=color)
            draw.text((bbox[0], bbox[1]), text, fill=(0, 0, 0))

        return im
