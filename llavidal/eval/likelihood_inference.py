import os
import sys
import argparse
import json
from tqdm import tqdm
from llavidal.eval.model_utils import initialize_model, load_video
from llavidal.likelihood_inference import llavidal_infer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True, default='')
    parser.add_argument('--qa_file', help='Path to the QA file containing video ids, questions, and answers.', required=True, default='')
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True, default='')
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True, default='')
    parser.add_argument("--model-name", type=str, required=True, default='')
    parser.add_argument("--conv-mode", type=str, required=False, default='llavidal_v1')
    parser.add_argument("--projection_path", type=str, required=True, default='')
    return parser.parse_args()

def save_to_json(output_dir, output_name, data):
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def run_inference(args):
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path)
    with open(args.qa_file) as file:
        qa_data = json.load(file)
    conv_mode = args.conv_mode
    correct_count = 0
    total = 0
    for key, sample in tqdm(qa_data.items()):
        video_id = sample['video_id']
        question = sample['question']   
        choices = sample['choices']
        ground_truth = sample['ground_truth']
        choices_text = ' '.join([f"{k}: {v}" for k, v in choices.items()])
        formatted_question = f"{question} Choices are {choices_text}"
        prompts = [f"{choices[f'{x}']}" for x in range(1, 5)]
        video_path = os.path.join(args.video_dir, os.path.basename(video_id))
        if os.path.exists(video_path):
            video_frames = load_video(video_path)
            prediction = llavidal_infer(video_frames, question, prompts, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len)
            predicted_answer = choices[f"{prediction + 1}"]
            print(ground_truth , predicted_answer)
            correct_count += ground_truth == predicted_answer
            total += 1
        print(f"Accuracy: {correct_count / total}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
