import os
import argparse
import json
from tqdm import tqdm
from llavidal.eval.model_utils import initialize_model, load_video
from llavidal.inference import llavidal_infer
from decord._ffi.base import DECORDError

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=True, default='llavidal_v1')
    parser.add_argument("--projection_path", type=str, required=True)
    return parser.parse_args()

def run_inference(args):
    """
    Run inference on a set of video files using the provided model.
    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    fixed_question = "Please describe the primary actions and interactions in the video, focusing on movements and the use of objects by any person or persons present."

    # Iterate over each video file in the directory
    for filename in tqdm(os.listdir(args.video_dir)):
        if any(filename.endswith(fmt) for fmt in video_formats):
            video_path = os.path.join(args.video_dir, filename)

            # Check if the video file exists before loading
            if os.path.exists(video_path):
                print(f"Processing video file: {video_path}")
                try:
                    video_frames = load_video(video_path)
                    # Run inference on the video and add the output to the list
                    output = llavidal_infer(video_frames, fixed_question, args.conv_mode, model,
                                                 vision_tower, tokenizer, image_processor, video_token_len)
                    output_list.append({
                        'video_name': filename,
                        'question': fixed_question,
                        'pred': output
                    })
                except DECORDError as e:
                    print(f"Decord error processing video file '{video_path}': {e}")
                except Exception as e:
                    print(f"Error processing video file '{video_path}': {e}")
            else:
                print(f"Video file not found: {video_path}")
                continue  # Skip this video and continue with the next

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)


























##################WRITTEN-CODE#########################################
# import os
# import argparse
# import json
# from tqdm import tqdm
# from llavidal.eval.model_utils import initialize_model, load_video
# from llavidal.inference import llavidal_infer

# def parse_args():
#     """
#     Parse command-line arguments.
#     """
#     parser = argparse.ArgumentParser()

#     # Define the command-line arguments
#     parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
#     parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
#     parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
#     parser.add_argument("--model-name", type=str, required=True)
#     parser.add_argument("--conv-mode", type=str, required=True, default='llavidal_v1')
#     parser.add_argument("--projection_path", type=str, required=True)

#     return parser.parse_args()

# def run_inference(args):
#     """
#     Run inference on a set of video files using the provided model.
#     Args:
#         args: Command-line arguments.
#     """
#     # Initialize the model
#     model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
#                                                                                         args.projection_path)
#     # Create the output directory if it doesn't exist
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)

#     output_list = []  # List to store the output results
#     video_formats = ['.mp4', '.avi', '.mov', '.mkv']
#     fixed_question = "Please describe the primary actions and interactions in the video, focusing on movements and the use of objects by any person or persons present."

#     # Iterate over each video file in the directory
#     for filename in tqdm(os.listdir(args.video_dir)):
#         if any(filename.endswith(fmt) for fmt in video_formats):
#             video_path = os.path.join(args.video_dir, filename)
#             # Check if the video file exists before loading
#             if os.path.exists(video_path):
#                 print(f"Processing video file: {video_path}") 
#                 video_frames = load_video(video_path)
#                 try:
#                     # Run inference on the video and add the output to the list
#                     output = llavidal_infer(video_frames, fixed_question, args.conv_mode, model,
#                                                  vision_tower, tokenizer, image_processor, video_token_len)
#                     output_list.append({
#                         'video_name': filename,
#                         'question': fixed_question,
#                         'pred': output
#                     })
#                 except Exception as e:
#                     print(f"Error processing video file '{filename}': {e}")
#             else:
#                 print(f"Video file not found: {video_path}")
#                 continue  # Skip this video and continue with the next

#     # Save the output list to a JSON file
#     with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
#         json.dump(output_list, file)

# if __name__ == "__main__":
#     args = parse_args()
#     run_inference(args)
