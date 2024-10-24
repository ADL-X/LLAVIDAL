import os
import argparse
import json
from tqdm import tqdm
import sys

import mcq_parsing_llm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videochatgpt_path', help='Directory where you cloned videochatgpt', required=True, default='/data/users/dreilly1/Video-ChatGPT/')
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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    conv_mode = args.conv_mode

    letter_lookup = {
        '1': 'A',
        '2': 'B',
        '3': 'C',
        '4': 'D',
        '5': 'E'
    }

    correct_count = 0
    total_count = 0

    iterator = tqdm(qa_data.items())
    for i, sample in iterator:
        video_id = sample['video_id']
        question = sample['question'] 
        choices = sample['choices']
        ground_truth = sample['ground_truth']

        letter_gt = None
        for k, v in choices.items():
            # check if the list v is equal to the list ground_truth
            if v == ground_truth:
                letter_gt = letter_lookup[k]
                break

        assert letter_gt is not None, f"Ground truth not found in choices for video '{video_id}'."
        qa_data[i]['ground_truth_letter'] = letter_gt

        # build the choices string
        choices_str = []
        for k, v in choices.items():
            letter = letter_lookup[k]
            choices_str.append(f"({letter}) {v}")

        choices_str = " ".join(choices_str)
        qa_data[i]['options_with_letter'] = choices_str

        formatted_question = f"{question}{choices_str}"
        video_path = os.path.join(args.video_dir, os.path.basename(video_id))

        if os.path.exists(video_path):
            video_frames = load_video(video_path)
            try:
                prediction = video_chatgpt_infer(video_frames, formatted_question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len)
                qa_data[i]['prediction'] = prediction

                prompt = mcq_parsing_llm.build_prompt(question, choices_str, prediction)
                letter_answer, llm_out = mcq_parsing_llm.parse_with_llama(prompt)
                qa_data[i]['parsed_answer_from_llm'] = letter_answer

                if letter_answer == letter_gt:
                    correct_count += 1                
            except Exception as e:
                print(f"Error processing video file '{video_id}': {e}")
        else:
            print(f"Video file '{video_id}' not found.")
            continue

        total_count += 1
        iterator.set_description(f"Accuracy: {correct_count / total_count * 100:.2f}%")

    print(f"Final Accuracy: {correct_count / total_count * 100:.8f}% if total_count > 0 else 0.00%")
    save_to_json(args.output_dir,f"{args.output_name}.json", qa_data)
    
if __name__ == "__main__":
    raise NotImplementedError("Use multiprocessing version")
    args = parse_args()
    sys.path.append(args.videochatgpt_path)

    from video_chatgpt.eval.model_utils import initialize_model, load_video
    from video_chatgpt.inference import video_chatgpt_infer

    run_inference(args)
