import os
import argparse
import json
from tqdm import tqdm
import re
import sys

import mcq_parsing_llm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videochatgpt_path', help='Directory where you cloned videochatgpt', required=True, default='/data/users/dreilly1/Video-ChatGPT/')
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True, default='')
    parser.add_argument('--qa_file', help='Path to the QA file containing video ids, questions, and answers.', required=False, default='/path/to/qa_file.json')
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=False, default='/path/to/output_dir')
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=False, default='output_file_name')
    parser.add_argument("--model-name", type=str, required=False, default='/path/to/model')
    parser.add_argument("--conv-mode", type=str, required=False, default='llavidal_v1')
    parser.add_argument("--projection_path", type=str, required=False, default='/path/to/projection.bin')
    parser.add_argument("--use-token-modality-prefix", help='Use token modality prefix for the model.', action='store_true')
    parser.add_argument("--use-string-modality-prefix", help='Use string modality prefix for the model.', action='store_true')
    parser.add_argument("--model-trained-with-base-videochatgpt", help='Model youre evaluating was trained with base videochatgpt code (changes how start/end tokens are loaded).', action='store_true')
    parser.add_argument("--debug", action='store_true', help='Debug mode.')
    return parser.parse_args()

def save_to_json(output_dir, output_name, data):
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def run_inference(args):
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path, args.use_token_modality_prefix, args.use_string_modality_prefix, args.model_trained_with_base_videochatgpt)

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
        '5': 'E',
        '6': 'F',
    }

    correct_count = 0
    total_count = 0

    iterator = tqdm(enumerate(qa_data), total=len(qa_data))
    for i, sample in iterator:
        video_id = sample['id']
        video_path = os.path.join(args.video_dir, video_id+'.mp4')
        question = sample['Q'] + ' The output should be the choice among one of the following options:'

        options = sample['Options']
        # split options string by any number followed by a period
        pattern = r"(?<!^)(?=\d+\.)"
        choices = [y.strip()[3:] for y in re.split(pattern, options)]
        choices_str = ' '.join([f"({letter_lookup[str(i+1)]}) {choices[i]}" for i in range(len(choices))])
        qa_data[i]['options_with_letter'] = choices_str

        ground_truth = sample['Ground_truth']
        letter_gt = None
        for j, v in enumerate(choices):
            # check if the list v is equal to the list ground_truth
            if v == ground_truth:
                letter_gt = letter_lookup[str(j+1)]
                break

        assert letter_gt is not None, f"Ground truth not found in options for question: {question}"
        qa_data[i]['ground_truth_letter'] = letter_gt

        formatted_question = f"{question} {choices_str}"

        if os.path.exists(video_path):
            try:
                video_frames = load_video(video_path)

                if video_frames is None:
                    print(f"Skipping video: {video_id}")
                    continue

                prediction = model_infer(video_frames, formatted_question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len)
                if args.debug:
                    print('================Question================')
                    print(formatted_question)
                    print('=================Answer=================')
                    print(prediction)
                    print('========================================')
                qa_data[i]['prediction'] = prediction

                prompt = mcq_parsing_llm.build_prompt(question, choices_str, prediction)
                letter_answer, llm_out = mcq_parsing_llm.parse_with_llama(prompt)
                qa_data[i]['parsed_answer_from_llm'] = letter_answer

                if letter_answer == letter_gt:
                    correct_count += 1
            except Exception as e:
                print(f"Error processing video file '{video_id}': {e}")
        else:
            print(f"Video file not found: {video_path}")
            continue

        total_count += 1
        iterator.set_description(f"Accuracy: {correct_count / total_count * 100:.2f}%")

    print(f"Final Accuracy: {correct_count / total_count * 100:.2f}% if total_count > 0 else 0.00%")

    # save .log file
    log_path = os.path.join(args.output_dir, f"{args.output_name}.log")
    if os.path.exists(args.output_dir):
        with open(log_path, 'a') as file:
            file.write("========================================\n")
            file.write(f"Final Accuracy (all processes): {correct_count / total_count * 100:.2f}%\n")
            file.write(f"Arguments: {args}\n")
    else:
        with open(log_path, 'w') as file:
            file.write(f"Final Accuracy (all processes): {correct_count / total_count * 100:.2f}%\n")
            file.write(f"Arguments: {args}\n")

    save_to_json(args.output_dir,f"{args.output_name}.json", qa_data)
if __name__ == "__main__":
    args = parse_args()

    # assert args.use_token_modality_prefix or args.use_string_modality_prefix, 'You must at least one of --use-token-modality-prefix or --use-string-modality-prefix. Choose one depending on the model being evaluated.'

    sys.path.append(args.videochatgpt_path)

    try:
        from video_chatgpt.eval.model_utils import initialize_model, load_video
        from video_chatgpt.inference import video_chatgpt_infer as model_infer
    except ImportError as e:
        from llavidal.eval.model_utils import initialize_model, load_video
        from llavidal.inference import llavidal_infer as model_infer

    run_inference(args)