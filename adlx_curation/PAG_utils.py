import numpy as np
import cv2

from einops import rearrange

def npy_to_keypoints(pose_path, frame_w, frame_h, clamp=False):
  np_skeleton = np.load(pose_path, allow_pickle=True).item()
  
  num_bodys = np.array(np_skeleton['nbodys']).max()
  keypoints_all = np.array([np_skeleton[f'rgb_body{i}'] for i in range(num_bodys)])
  njts = np_skeleton['njoints']

  B = keypoints_all.shape[0]
  T = keypoints_all.shape[1]
  J = keypoints_all.shape[2]

  keypoints_all = rearrange(keypoints_all, 'b f n c -> f (b n) c', b=B, f=T, n=J, c=2)

  if num_bodys > 1:
    for frm_idx, frm_body_count in enumerate(np_skeleton['nbodys']):
      # Keypoint matrix contains all bodies in the video. Some bodies
      # are not contained in every frame and those bodies keypoints are
      # set to 0's. Here we set that bodies keypoints to the 1st bodies keypoints
      for body_idx in range(num_bodys-1, frm_body_count-1, -1):
        # print(f'\tsetting body {body_idx} to 1st body kpts')
        keypoints_all[frm_idx, body_idx*J:(body_idx+1)*J, :] = keypoints_all[frm_idx, :25, :]

  # Useful for get_frame function for debugging
  if clamp:
    keypoints_all = clamp_keypoints(keypoints_all, frame_w, frame_h)

  return keypoints_all

def clamp_keypoints(keypoints, frame_w, frame_h):
  '''
  In some cases, the keypoints can be nan, or can be outside of the frame.
  -This function replaces nan keypoints with its closest neighbor
  -This function bounds keypoints to the frame boundaries
  '''
  clipped_kpts = keypoints.copy()

  mask = np.isnan(clipped_kpts)
  clipped_kpts[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), clipped_kpts[~mask])

  clipped_kpts[:, :, 0] = np.clip(clipped_kpts[:, :, 0], 0, frame_w)
  clipped_kpts[:, :, 1] = np.clip(clipped_kpts[:, :, 1], 0, frame_h)

  return clipped_kpts


def get_coords_from_pose(keypoints, frame_idx, njts=None):
  '''
  Returns properties of a bounding box spanning the poses
  '''
  if njts is None:
      njts = keypoints.shape[1]
  x = np.reshape(keypoints[frame_idx], (njts, 1, 2)).astype(int)
  x,y,w,h = cv2.boundingRect(x)

  center = (x+w//2,y+h//2)

  return x, y, w, h, center

def get_bbox_const_sz(keypoints, frame_idx, box_w, box_h, njts=13):
  tlx, tly, w, h, center = get_coords_from_pose(keypoints, frame_idx, njts)

  assert (box_h % 2 == 0) and (box_w % 2 == 0), 'Crop size must be even'

  top_left_corner = (
      center[0] - box_w//2,
      center[1] - box_h//2
  )

  bot_right_corner = (
      center[0] + box_w//2,
      center[1] + box_h//2
  )

  return top_left_corner, bot_right_corner

def get_bbox_pose_centered(keypoints, frame_idx):
  tlx, tly, w, h, center = get_coords_from_pose(keypoints, frame_idx)

  top_left_corner = (
    tlx,
    tly
  )

  bot_right_corner = (
      tlx + w,
      tly + h
  )

  return top_left_corner, bot_right_corner

def get_bbox(keypoints, frame_idx, slack=64):
  '''
  Returns properties of a bounding box spanning the poses with a given slack.
  Box is defined by top left and bottom right corner coordinates
  '''
  tlx, tly, w, h, center = get_coords_from_pose(keypoints, frame_idx)
  box_w = w + slack
  box_h = h + slack

  # assert (box_h % 2 == 0) and (box_w % 2 == 0), 'Crop size must be even'

  top_left_corner = (
      center[0] - box_w//2,
      center[1] - box_h//2
  )

  bot_right_corner = (
      center[0] + box_w//2,
      center[1] + box_h//2
  )

  return top_left_corner, bot_right_corner

def process_frame(frame, keypoints, tlc, brc, new_shape=None):
  '''
  Crop a frame given a bounding box. Return the cropped frame and adjusted keypoints
  ** Arguments **
  frame : np.ndarray
    The frame to crop
  keypoints : np.ndarray
    Keypoints associated with the frame
  tlc : tuple[int]
    Top Left Corner of bounding box
  brc : tuple[int]
    Bottom Right Corner of bounding box
  new_shape : tuple[int]
    Shape to resize the crop to
  '''
  frame_w, frame_h = frame.shape[1], frame.shape[0]
  box_w, box_h = (brc[0] - tlc[0]), (brc[1] - tlc[1])

  bbox_contained_in_frame = True
  if (tlc[0] < 0) or (tlc[1] < 0) or (brc[0] >= frame_w) or (brc[1] >= frame_h):
    bbox_contained_in_frame = False

  # pad image if bbox extends past frame boundaries
  if not bbox_contained_in_frame:
    bsz = (box_h, box_h, box_w, box_w) # border size (top, bot, left, right). We can always assume top=bot and left=right
    frame = cv2.copyMakeBorder(frame, *bsz, cv2.BORDER_CONSTANT)
  else:
    bsz = (0, 0, 0, 0)

  # adjust top-left-corner and bottom-right-corner to match padded image
  tlc = tlc[0] + bsz[2], tlc[1] + bsz[0]
  brc = brc[0] + bsz[2], brc[1] + bsz[0]

  frame = frame[tlc[1] : brc[1], 
                  tlc[0] : brc[0]]

  # adjust frame keypoints to match padded image
  keypoints[:, 0] += bsz[2] - (tlc[0])
  keypoints[:, 1] += bsz[0] - (brc[1] - box_h)

  if new_shape:
    cur_shape = frame.shape
    frame = cv2.resize(frame, new_shape)

    x_ratio, y_ratio = (new_shape[0] / cur_shape[1]), (new_shape[1] / cur_shape[0])

    keypoints[:, 0] *= x_ratio
    keypoints[:, 1] *= y_ratio

  return frame, keypoints