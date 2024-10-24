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
    parser.add_argument('--qa_file', help='Path to the QA file containing video ids, questions, and answers.', required=False, default='/path/to/qa_file.json')
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=False, default='/path/to/output_dir')
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=False, default='output_file_name')
    parser.add_argument("--model-name", type=str, required=False, default='/path/to/model')
    parser.add_argument("--conv-mode", type=str, required=False, default='llavidal_v1')
    parser.add_argument("--projection_path", type=str, required=False, default='/path/to/projection.bin')
    parser.add_argument("--use-token-modality-prefix", help='Use token modality prefix for the model.', action='store_true')
    parser.add_argument("--use-string-modality-prefix", help='Use string modality prefix for the model.', action='store_true')
    parser.add_argument("--debug", action='store_true', help='Debug mode.')
    return parser.parse_args()

def save_to_json(output_dir, output_name, data):
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def run_inference(args):
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path, args.use_token_modality_prefix, args.use_string_modality_prefix)

    with open(args.qa_file) as file:
        qa_data = json.load(file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    conv_mode = args.conv_mode

    correct_count = 0
    total_count = 0

    iterator = tqdm(qa_data.items())
    for i, sample in iterator:
        video_id = sample['video_id']
        start_frame = sample['start_frame']
        end_frame  = sample['end_frame']
        video_id = f'{video_id}_{start_frame}_{end_frame}.mp4'
        video_path = os.path.join(args.video_dir, video_id)
        question = sample['question'] + '. The output should be the choice among one of the following choices.'
        choices = sample['choices']

        ground_truth = choices[sample['answer']]

        # Format choices as a string
        if isinstance(choices, dict):
            choices_str = ' '.join([f'({k}) {choice}' for k, choice in choices.items()])
        elif isinstance(choices, list):
            choices_str = ' '.join(choices)
        else:
            raise ValueError(f"Unexpected format for choices: {choices}")

        formatted_question = f"{question} Choices are {choices_str}"

        if os.path.exists(video_path):
            video_frames = load_video(video_path)
            if video_frames is None:
                print(f"Skipping video: {video_id}")
                continue

            try:
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

                if letter_answer == sample['answer']:
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

    sys.path.append(args.videochatgpt_path)

    try:
        from video_chatgpt.eval.model_utils import initialize_model, load_video
        from video_chatgpt.inference import video_chatgpt_infer as model_infer
    except ImportError as e:
        from llavidal.eval.model_utils import initialize_model, load_video
        from llavidal.inference import llavidal_infer as model_infer

    run_inference(args)