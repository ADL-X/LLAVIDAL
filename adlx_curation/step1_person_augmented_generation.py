import matplotlib.pyplot as plt
import numpy as np
import simplejson
import glob
import copy
import cv2
import os

from PAG_utils import npy_to_keypoints, clamp_keypoints, get_bbox, process_frame

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ntu_data_path', type=str, required=True, help='Absolute path to NTU120 directory. Should contain a directory "rgb" containing all of the videos')
parser.add_argument('--new_dataset_name', type=str, required=True,
        help='Name to append to NTU. After generating, a directory will be created in ntu_data_path: {ntu_data_path}/NTU_{new_dataset_name}'
)
parser.add_argument('--slack', type=int, required=False, default=64)
parser.add_argument('--new_shape', nargs='+', type=int, required=True, help='Tuple giving shape to resize frame to after cropping. pass like "--new_shape W H"')

args = parser.parse_args()
if args.ntu_data_path[-1] == '/':
  args.ntu_data_path = args.ntu_data_path[:-1]

args.new_shape = tuple(args.new_shape)


def generate_cropped_dataset(ntu_subset_path, new_shape):
    '''
    ntu_subset_path : str
        Path to NTU. e.g. '/data/ntu/NTU'
    '''
    all_videos = glob.glob(f'{ntu_subset_path}/rgb/*.avi')

    save_dir, ntu_subset = os.path.split(ntu_subset_path)
    os.makedirs(f'{save_dir}/{ntu_subset}_{args.new_dataset_name}/rgb', exist_ok=True)
    os.makedirs(f'{save_dir}/{ntu_subset}_{args.new_dataset_name}/skeletons', exist_ok=True)

    print(f'\tProcessing data at {ntu_subset_path}. Saving at {save_dir}/{ntu_subset}_{args.new_dataset_name}')

    write_shape = new_shape

    for i, vid_path in enumerate(all_videos):
        if i % (len(all_videos) // 10) == 0:
            print(f'\tProcessing video {i}/{len(all_videos)}')

        # Load video, initialize video writer
        video_path_stem, video_identifier = os.path.split(vid_path)
        video_identifier = video_identifier[:-8] # clip '_rgb.avi'

        # skip video if it is missing skeleton
        if not os.path.exists(f'{ntu_subset_path}/skeletons/{video_identifier}.skeleton.npy'):
            continue

        if os.path.exists(f'{save_dir}/{ntu_subset}_{args.new_dataset_name}/rgb/{video_identifier}_rgb.avi'):
            print('skipping {video_identifier} (already exists)')
            continue

        cap = cv2.VideoCapture(vid_path)
        w, h, frame_rate, num_frames = int(cap.get(3)), int(cap.get(4)), int(cap.get(5)), int(cap.get(7))

        writer = cv2.VideoWriter(f'{save_dir}/{ntu_subset}_{args.new_dataset_name}/rgb/{video_identifier}_rgb.avi',
                                 cv2.VideoWriter_fourcc(*'FMP4'),
                                 frame_rate, write_shape, True
        )

        # Load keypoints
        pose_path = f'{ntu_subset_path}/skeletons/{video_identifier}.skeleton.npy'
        np_skeleton = np.load(pose_path, allow_pickle=True).item() # used when re-saving keypoints after shifting them
        number_of_bodies_in_video = np.array(np_skeleton['nbodys']).max()
        njts = np_skeleton['njoints']
        
        kpts = npy_to_keypoints(pose_path, frame_h=h, frame_w=w)
        kpts = clamp_keypoints(kpts, frame_h=h, frame_w=w) # used to generate the multi-person crops. Will be shape (n_frame, bodies*njts, 2)
        num_frames_with_kpts = kpts.shape[0]

        # Generate video and update keypoints frame-by-frame
        error_printed = False
        for frm_num in range(0, num_frames_with_kpts):
            try:
                number_of_bodies_in_frame = np_skeleton['nbodys'][frm_num]
            except IndexError as e:
                if not error_printed:
                    print(f'Error indexing nbodys(frame {frm_num}) in {vid_path}: nbodys shape: {len(np_skeleton["nbodys"])} - kpts shape: {kpts.shape}')
                    error_printed = True

                number_of_bodies_in_frame = 1

            ret, frame = cap.read()

            tlc, brc = get_bbox(kpts, frm_num, slack=args.slack)
            new_frm, new_kpts = process_frame(frame, kpts[frm_num], tlc, brc, new_shape)

            writer.write(new_frm)

            # Update keypoints associated with the video. In npy_to_keypoints if a body leaves the frame, that bodies keypoints are
            # set to the keypoints of the first body (which is always in the frame) to make processing easier. 
            # We dont want to save these duplicated keypoints, so we save the original keypoints for bodies not in the frame
            for body_idx in range(number_of_bodies_in_video):
                # the first body will always be in the frame, always update its keypoints
                if body_idx == 0:
                    np_skeleton['rgb_body0'][frm_num] = new_kpts[body_idx*njts:(body_idx+1)*njts]
                else:
                    if number_of_bodies_in_frame > body_idx: # this body is in the frame
                        np_skeleton[f'rgb_body{body_idx}'][frm_num] = new_kpts[body_idx*njts:(body_idx+1)*njts]
                    else: # make no changes if the body is not contained in the frame
                        pass


        writer.release()
        cap.release()

        # Save updated keypoints and indicate that all body keypoints have been updated (in the past only rgb_body0 was updated)
        np_skeleton['all_bodies_updated'] = True
        np.save(f'{save_dir}/{ntu_subset}_{args.new_dataset_name}/skeletons/{video_identifier}.skeleton.npy', np.array(np_skeleton))


input(f'Generating cropped dataset for NTU at path: {args.ntu_data_path}. Cropping parameters new_shape: {args.new_shape}. New dataset name {args.new_dataset_name} (will save to {args.ntu_data_path}/{args.new_dataset_name}).\nPress enter to continue or Ctrl + Z to quit!')

ntu_path = f'{args.ntu_data_path}'
print('\nGenerating dataset from ({ntu_path})')
generate_cropped_dataset(ntu_path, new_shape=args.new_shape)