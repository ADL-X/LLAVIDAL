import argparse, decord, tqdm, json, glob, os
from PIL import Image

from moviepy.editor import VideoFileClip, concatenate_videoclips

parser = argparse.ArgumentParser(description='Reconstruct NTU videos from cropped frames')
parser.add_argument('--cropped_ntu_dir', type=str, help='Directory containing cropped NTU120 video dataset')
parser.add_argument('--video_mapping_json', type=str, help='Path to video mapping JSON file')
parser.add_argument('--save_dir', type=str)
args = parser.parse_args()

cropped_ntu_dir = args.cropped_ntu_dir
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

all_video_mappings = json.load(open(args.video_mapping_json, 'r'))

iterator = tqdm.tqdm(all_video_mappings.items(), total=len(all_video_mappings))

for stitched_video_name, constituent_videos in iterator:
    stitched_video_path = os.path.join(save_dir, stitched_video_name + '.mp4')

    clips_for_stitched_video = []

    for cons_vid in constituent_videos:
        cons_vid_path = os.path.join(cropped_ntu_dir, cons_vid)
        if not os.path.exists(cons_vid_path):
            print('Missing:', cons_vid_path)
            continue
        
        try:
            clip = VideoFileClip(cons_vid_path)
            clips_for_stitched_video.append(clip)
        except Exception as e:
            print('Error:', cons_vid_path, e)
    
    if len(clips_for_stitched_video) != 0:
        try:
            stitched_video = concatenate_videoclips(clips_for_stitched_video)
            stitched_video.write_videofile(stitched_video_path, codec='libx264', verbose=False, logger=None)
            stitched_video.close()
        except Exception as e:
            print('Error:', stitched_video_path, e)