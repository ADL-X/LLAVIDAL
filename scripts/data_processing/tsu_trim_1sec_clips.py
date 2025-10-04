"""
Script to process Toyota Smarthome Untrimmed (Dai et al.) and obtain videos required for TSU-Descriptions evaluation.

Generate the data used for evaluation on TSU descriptions (i.e., generate 1 second clips of each untrimmed video)

Usage:
    python tsu_trim_1sec_clips.py --tsu_videomp4 /path/to/smarthome_untrimmed/Videos_MP4 --output_vid_dir /path/to/save/TSU_Descriptions_videos
"""
import os
import subprocess
import datetime
import argparse

def slice_video(video_path, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Get the duration of the video
    duration_output = subprocess.check_output(['ffprobe', '-i', video_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0'])
    duration = float(duration_output.decode().strip())

    # Calculate the number of clips
    num_clips = int(duration // 60) + 1

    log = f"Original Video: {video_path}\n"
    log += f"Total Duration: {datetime.timedelta(seconds=duration)}\n"
    log += f"Number of Clips: {num_clips}\n\n"

    # Slice the video into 1-minute clips
    for i in range(num_clips):
        start_time = i * 60
        end_time = min((i + 1) * 60, duration)

        clip_name = f"{video_name}_clip{i+1}.mp4"
        clip_path = os.path.join(output_dir, clip_name)

        # Use ffmpeg to slice the video clip
        subprocess.call(['ffmpeg', '-i', video_path, '-ss', str(start_time), '-t', str(end_time - start_time), '-c', 'copy', clip_path])

        log += f"Clip {i+1}:\n"
        log += f"  Name: {clip_name}\n"
        log += f"  Start Time: {datetime.timedelta(seconds=start_time)}\n"
        log += f"  End Time: {datetime.timedelta(seconds=end_time)}\n"
        log += f"  Duration: {datetime.timedelta(seconds=end_time - start_time)}\n\n"

    # Save the log
    log_path = os.path.join(output_dir, f"{video_name}_log.txt")
    with open(log_path, 'w') as f:
        f.write(log)

    print(f"Video slicing completed for {video_name}. Log saved to: {log_path}")

# Parse arguments
parser = argparse.ArgumentParser(description='Batch process videos and create 1 second clips')
parser.add_argument('--tsu_videomp4', type=str, required=True, help='Path to the directory containing TSU untrimmed .mp4 videos')
parser.add_argument('--output_vid_dir', type=str, required=True, help='Directory to save the output video clips and logs')
args = parser.parse_args()

test_participant_ids = [2, 10, 11, 14, 16, 18, 20] # participant IDs of participants in the test set of TSU

for filename in os.listdir(args.tsu_videomp4):
    if filename.endswith(".mp4"):
        video_path = os.path.join(args.tsu_videomp4, filename)
        slice_video(video_path, args.output_vid_dir)
 