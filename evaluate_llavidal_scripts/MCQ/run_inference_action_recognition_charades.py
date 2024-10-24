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
    parser.add_argument('--qa_file', help='Path to the QA file containing questions and answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default='llavidal_v1')
    parser.add_argument("--projection_path", type=str, required=True)
    return parser.parse_args()

def save_to_json(output_dir, output_name, data):
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def parse_options(options_str):
    options_dict = {}
    pattern = r"(\d+)\.\s*\[(.*?)\]"
    matches = re.findall(pattern, options_str)
    for match in matches:
        key, value = match
        options_dict[key] = [item.strip().strip("'") for item in value.split(',')]
    return options_dict

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

    iterator = tqdm(enumerate(qa_data), total=len(qa_data))
    for i, sample in iterator:
        video_path = os.path.join(args.video_dir, f"{sample['id']}.mp4")
        question = sample['Q']
        options = parse_options(sample['Options']) # MCQ choices were saved as a string representation of a list in the json file, have to parse it back to a list
        ground_truth = sample['Ground Truth']

        # Ground truth is a list, have to check which MCQ choice corresponds to it
        letter_gt = None
        for k, v in options.items():
            # check if the list v is equal to the list ground_truth
            if v == ground_truth:
                letter_gt = letter_lookup[k]
                break

        assert letter_gt is not None, f"Ground truth not found in options for question: {question}"
        qa_data[i]['ground_truth_letter'] = letter_gt

        # build the choices string
        choices_str = []
        for k, v in options.items():
            letter = letter_lookup[k]
            choices_str.append(f"({letter}) {', '.join(v)}")

        choices_str = " ".join(choices_str)
        qa_data[i]['options_with_letter'] = choices_str

        full_question = f"{question} The output should be the choice among one of the following choices. Choices are {choices_str}"

        if os.path.exists(video_path):
            video_frames = load_video(video_path)
            if video_frames is None:
                print(f"Skipping video: {video_path}")
                continue
            try:
                prediction = video_chatgpt_infer(video_frames, full_question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len)
                qa_data[i]['prediction'] = prediction

                prompt = mcq_parsing_llm.build_prompt(question, choices_str, prediction)
                letter_answer, llm_out = mcq_parsing_llm.parse_with_llama(prompt)
                qa_data[i]['parsed_answer_from_llm'] = letter_answer

                # print(f'Question: {question}')
                # print(f'Choices: {choices_str}')
                # print(f'Prediction: {prediction}')
                # print(f'LLM Output: {llm_out}')
                # print(f'Answer: {letter_gt}')
                # print(f'Ground truth: {ground_truth}')
                # print(f'Letter Answer from LLM: {letter_answer}\n')

                if letter_answer == letter_gt:
                    correct_count += 1 
            except Exception as e:
                print(f"Error processing video file '{video_path}': {str(e)}")
        else:
            print(f"Video file not found: {video_path}")
            continue

        total_count += 1
        iterator.set_description(f"Accuracy: {correct_count / total_count * 100:.2f}%")

    print(f"Final Accuracy: {correct_count / total_count * 100:.2f}% if total_count > 0 else 0.00%")
    save_to_json(args.output_dir,f"{args.output_name}.json", qa_data)
if __name__ == "__main__":
    raise NotImplementedError("Use multiprocessing version")
    args = parse_args()

    sys.path.append(args.videochatgpt_path)

    from video_chatgpt.eval.model_utils import initialize_model, load_video
    from video_chatgpt.inference import video_chatgpt_infer

    run_inference(args)