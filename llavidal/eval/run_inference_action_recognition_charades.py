import os
import argparse
import json
from tqdm import tqdm
from llavidal.eval.model_utils import initialize_model, load_video
from llavidal.inference import llavidal_infer

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
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_list = []
    conv_mode = args.conv_mode
    correct_count = 0
    total_count = 0
    
    for sample in tqdm(qa_data):
        video_id = sample['id']
        question = sample['Q'] 
        choices = sample['Options'] + '. The output should be the choice among one of the following choices and return the option either 1 or 2 or 3 or 4.'
        ground_truth = sample['Ground Truth']
        
        # Properly format choices
        choices_list = []
        options = choices.split('  ')
        for option in options:
            option_index, option_values = option.split('. ', 1)
            choices_list.append(f"{option_index.strip()}: {option_values.strip()}")
        
        choices_text = ' '.join(choices_list)
        formatted_question = f"{question} Choices are {choices_text}"
        #breakpoint()
        video_path = os.path.join(args.video_dir, f"{video_id}.mp4")
        
        if os.path.exists(video_path):
            video_frames = load_video(video_path)
            try:
                prediction = llavidal_infer(video_frames, formatted_question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len)
                sample['prediction'] = prediction
                if prediction == ground_truth:
                    correct_count += 1
                total_count += 1
                output_list.append({'video_id': video_id, 'ground_truth': ground_truth, 'prediction': prediction})
                save_to_json(args.output_dir, args.output_name, {
                    'results': output_list,
                    'accuracy': correct_count / total_count if total_count > 0 else 0
                })
            except Exception as e:
                print(f"Error processing video file '{video_id}': {e}")
    
    print(f"Final Accuracy: {correct_count / total_count * 100:.2f}% if total_count > 0 else 0.00%")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
