"""
Script to process Toyota Smarthome Untrimmed (Dai et al.) and obtain videos required for TSU-TC evaluation.

Generate the data used for evaluation on TSU-TC (i.e., trim the action segments from the untrimmed videos)

Usage:
    python tsu_trim_actions.py --csv_root_dir /path/to/smarthome_untrimmed/Annotation --video_root_dir /path/to/smarthome_untrimmed/Videos_MP4 --output_vid_dir /path/to/save/TSU_TC_videos --workers 4
"""
import json
import csv
import subprocess
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import glob
import logging
from datetime import datetime

@dataclass
class Action:
    event: str
    start_frame: int
    end_frame: int

def get_video_info(video_path: str) -> Dict[str, float]:
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,duration',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    num, den = map(int, info['streams'][0]['r_frame_rate'].split('/'))
    return {'fps': num / den}

def create_clip(video_path: str, clip: Dict, fps: float, output_dir: str) -> None:
    clip_num = clip["clip_number"]
    start_time = clip["start_frame"] / fps
    duration = (clip["end_frame"] - clip["start_frame"]) / fps
    output_path = os.path.join(output_dir, f"clip_{clip_num}.mp4")
    
    cmd = [
        'ffmpeg',
        '-y',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '28',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-ac', '2',
        '-map_metadata', '-1',
        '-movflags', '+faststart',
        '-threads', '0',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return clip_num
    except subprocess.CalledProcessError:
        return None

def extract_clips_parallel(video_path: str, clips: List[Dict], fps: float, output_dir: str, max_workers: int = 4):
    os.makedirs(output_dir, exist_ok=True)
    
    ffmpeg_clips = [{
        "clip_number": clip["clip_number"],
        "start_frame": clip["start_frame"],
        "end_frame": clip["end_frame"]
    } for clip in clips]
    
    with tqdm(total=len(clips), desc="Extracting clips") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    create_clip, video_path, clip, fps, output_dir
                ): clip["clip_number"] for clip in ffmpeg_clips
            }
            
            for future in as_completed(futures):
                clip_num = future.result()
                if clip_num:
                    pbar.update(1)
                    pbar.set_description(f"Processed clip {clip_num}")

def setup_logging(output_dir: str):
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'processing_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('batch_processor')

def find_video_file(csv_name: str, video_root_dir: str) -> str:
    video_name = csv_name.replace('.csv', '.mp4')
    for root, _, files in os.walk(video_root_dir):
        if video_name in files:
            return os.path.join(root, video_name)
    return None

def read_action_data(csv_file: str) -> List[Action]:
    actions = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f, fieldnames=['event', 'start_frame', 'end_frame'])
        next(reader)  # Skip header
        for row in reader:
            try:
                actions.append(Action(
                    event=row['event'],
                    start_frame=int(row['start_frame']),
                    end_frame=int(row['end_frame'])
                ))
            except ValueError as e:
                print(f"Skipping invalid row: {row}. Error: {e}")
    return actions

def save_formatted_json(clips: List[Dict], fps: float, output_json: str):
    formatted_clips = []
    
    for clip in clips:
        formatted_clip = {
            "clip_number": clip["clip_number"],
            "time_info": {
                "start_frame": clip["start_frame"],
                "end_frame": clip["end_frame"],
                "duration_frames": clip["end_frame"] - clip["start_frame"],
                "duration_seconds": round((clip["end_frame"] - clip["start_frame"]) / fps, 2)
            },
            "actions_info": {
                "num_actions": len(clip["actions"]),
                "actions": [{
                    "event": action["event"],
                    "start_frame": action["start_frame"],
                    "end_frame": action["end_frame"],
                    "duration": round((action["end_frame"] - action["start_frame"]) / fps, 2)
                } for action in clip["actions"]]
            }
        }
        formatted_clips.append(formatted_clip)

    formatted_data = {
        "metadata": {
            "total_clips": len(clips),
            "fps": fps,
            "total_actions": sum(len(clip["actions"]) for clip in clips)
        },
        "clips": formatted_clips
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)

def main(video_path: str, input_csv: str, output_dir: str = "clips", 
         output_json: str = "video_clips.json", max_workers: int = 4):
    print("Getting video information...")
    video_info = get_video_info(video_path)
    fps = video_info['fps']
    
    print("Reading actions from CSV...")
    actions = read_action_data(input_csv)
    
    print("Creating clips...")
    clips = []
    current_clip = []
    
    for action in tqdm(actions, desc="Processing actions"):
        if len(current_clip) == 0:
            current_clip = [{
                "event": action.event,
                "start_frame": action.start_frame,
                "end_frame": action.end_frame
            }]
        else:
            duration = (action.end_frame - current_clip[0]["start_frame"]) / fps
            if len(current_clip) >= 5 or duration > 60:
                if len(current_clip) >= 4:
                    clips.append({
                        "start_frame": current_clip[0]["start_frame"],
                        "end_frame": current_clip[-1]["end_frame"],
                        "actions": current_clip.copy()
                    })
                current_clip = [{
                    "event": action.event,
                    "start_frame": action.start_frame,
                    "end_frame": action.end_frame
                }]
            else:
                current_clip.append({
                    "event": action.event,
                    "start_frame": action.start_frame,
                    "end_frame": action.end_frame
                })
    
    if len(current_clip) >= 4:
        clips.append({
            "start_frame": current_clip[0]["start_frame"],
            "end_frame": current_clip[-1]["end_frame"],
            "actions": current_clip
        })
    
    for i, clip in enumerate(clips, 1):
        clip["clip_number"] = i
    
    print(f"Saving metadata to {output_json}...")
    save_formatted_json(clips, fps, output_json)
    
    print("Extracting video clips...")
    extract_clips_parallel(video_path, clips, fps, output_dir, max_workers)
    
    return clips

def process_participant_folder(participant_dir: str, video_root_dir: str, output_base_dir: str, max_workers: int = 4):
    logger = logging.getLogger('batch_processor')
    
    participant_name = os.path.basename(participant_dir)
    participant_output_dir = os.path.join(output_base_dir, participant_name)
    os.makedirs(participant_output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(participant_dir, "*.csv"))
    
    logger.info(f"Processing participant: {participant_name}")
    logger.info(f"Found {len(csv_files)} CSV files")
    
    summary = {
        "participant": participant_name,
        "total_files": len(csv_files),
        "processed_files": [],
        "failed_files": []
    }
    
    for csv_file in csv_files:
        csv_name = os.path.basename(csv_file)
        video_file = find_video_file(csv_name, video_root_dir)
        
        if video_file:
            logger.info(f"Processing: {csv_name}")
            try:
                video_output_dir = os.path.join(participant_output_dir, csv_name.replace('.csv', ''))
                os.makedirs(video_output_dir, exist_ok=True)
                
                clips = main(
                    video_path=video_file,
                    input_csv=csv_file,
                    output_dir=os.path.join(video_output_dir, 'clips'),
                    output_json=os.path.join(video_output_dir, 'metadata.json'),
                    max_workers=max_workers
                )
                
                summary["processed_files"].append({
                    "csv_file": csv_name,
                    "video_file": os.path.basename(video_file),
                    "num_clips": len(clips)
                })
                
                logger.info(f"Successfully processed {csv_name}")
            except Exception as e:
                error_msg = f"Error processing {csv_name}: {str(e)}"
                logger.error(error_msg)
                summary["failed_files"].append({
                    "csv_file": csv_name,
                    "video_file": os.path.basename(video_file),
                    "error": str(e)
                })
        else:
            error_msg = f"No matching video file found for {csv_name}"
            logger.warning(error_msg)
            summary["failed_files"].append({
                "csv_file": csv_name,
                "error": "No matching video file found"
            })
    
    summary_file = os.path.join(participant_output_dir, 'processing_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary

def batch_process(csv_root_dir: str, video_root_dir: str, output_base_dir: str, max_workers: int = 4):
    logger = setup_logging(output_base_dir)
    
    logger.info("Starting batch processing")
    logger.info(f"CSV root directory: {csv_root_dir}")
    logger.info(f"Video root directory: {video_root_dir}")
    
    # Changed the pattern from "p*" to "P*" to match your folder structure
    participant_dirs = [d for d in glob.glob(os.path.join(csv_root_dir, "P*")) if os.path.isdir(d)]
    
    # Add debug logging
    logger.info(f"Searching in directory: {csv_root_dir}")
    logger.info(f"Found directories: {[os.path.basename(d) for d in participant_dirs]}")
    
    if not participant_dirs:
        logger.error(f"No participant folders found in {csv_root_dir}")
        logger.info("Please check if the path is correct and contains P* folders")
        return
    
    logger.info(f"Found {len(participant_dirs)} participant folders")
    
    batch_summary = {
        "start_time": datetime.now().isoformat(),
        "total_participants": len(participant_dirs),
        "participants": []
    }
    
    for participant_dir in participant_dirs:
        summary = process_participant_folder(participant_dir, video_root_dir, output_base_dir, max_workers)
        batch_summary["participants"].append(summary)
    
    batch_summary["end_time"] = datetime.now().isoformat()
    
    summary_file = os.path.join(output_base_dir, 'batch_processing_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=4)
    
    logger.info("Batch processing complete")
    logger.info(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Batch process videos and create clips based on action sequences')
    parser.add_argument('--csv_root_dir', help='Root directory containing participant folders with CSV files')
    parser.add_argument('--video_root_dir', help='Root directory containing video files')
    parser.add_argument('--output_vid_dir', help='Base output directory for results')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers for video processing')
    args = parser.parse_args()
    
    os.makedirs(args.output_vid_dir, exist_ok=True)

    batch_process(args.csv_root_dir, args.video_root_dir, args.output_vid_dir, args.workers)