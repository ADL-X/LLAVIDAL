import argparse, datetime, random, json, sys, os
from tqdm import tqdm
import numpy as np

import torch.distributed as dist
import torch

import warnings
warnings.filterwarnings("ignore")

import mcq_parsing_llm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", help='Path to the LLaVA-7B-Lightening-v1-1 directory containing base weights of the model.', type=str, required=True)
    parser.add_argument("--proj_weight_path", help='Path to the .bin file containing projection weights of our model.', type=str, required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True, default='')
    parser.add_argument('--qa_file', help='Path to the QA file containing questions and answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of new tokens.')
    parser.add_argument("--debug", action='store_true', help='Debug mode.')
    parser.add_argument("--seed", type=int, default=127, help='Random seed.')
    return parser.parse_args()

def main():
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=120))
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #### Start of run_inference code ####
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.base_model_path, args.proj_weight_path)
    conv_mode = 'llavidal_v1'

    with open(args.qa_file) as file:
        qa_data = json.load(file)
    qa_data = split_data(qa_data, world_size, global_rank)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    correct_count = 0
    total_count = 0

    if not args.debug or local_rank == 0:
        iterator = tqdm(enumerate(qa_data), total=len(qa_data))
    else:
        iterator = enumerate(qa_data)

    for i, sample in iterator:
        total_count += 1

        video_path = os.path.join(args.video_dir, sample['video_filename'])
        
        if not os.path.exists(video_path):
            print(f"Video file '{video_path}' does not exist.")
            continue

        video_frames = load_video(video_path)

        question = sample['question']
        choices = sample['answer_choices']
        ground_truth_letter = sample['ground_truth_letter']

        # build the choices string
        choices_str = " ".join(f'({k}) {v}' for k, v in choices.items())
        qa_data[i]['question_to_llm'] = choices_str

        full_question = f"{question} The output should be the choice among one of the following choices. Choices are {choices_str}"

        try:
            prediction = model_infer(video_frames, full_question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, max_new_tokens=args.max_new_tokens)
            qa_data[i]['prediction'] = prediction

            prompt = mcq_parsing_llm.build_prompt(question, choices_str, prediction)
            parsed_letter_answer_from_llm, llm_out = mcq_parsing_llm.parse_with_llama(prompt)
            qa_data[i]['parsed_answer_from_llm'] = parsed_letter_answer_from_llm

            if args.debug and local_rank == 0:
                print(f"\n{'='*16} Question {'='*16}")
                print(full_question)
                print(f"\n{'='*16} Answer from LLM {'='*16}")
                print(prediction)
                print(f"\n{'='*16} Predicted/true ans {'='*16}")
                print(f'{parsed_letter_answer_from_llm} / {ground_truth_letter}\n')
            
            if parsed_letter_answer_from_llm == ground_truth_letter:
                correct_count += 1 
        except Exception as e:
            print(f"Error processing video file '{video_path}': {str(e)}")
            continue

        if not args.debug or local_rank == 0:
            iterator.set_description(f"{args.output_name} (process {local_rank}) Accuracy: {correct_count / total_count * 100:.2f}%")

    print(f"Final Accuracy (process {local_rank}): {correct_count / total_count * 100:.2f}")
    #### End of run_inference code ####

    del model
    torch.cuda.empty_cache()

    results = [correct_count, total_count, qa_data]

    gathered_results = [None for _ in range(dist.get_world_size())]
    dist.gather_object(
        obj=results,
        object_gather_list=gathered_results if global_rank == 0 else None,
        dst=0,
    )

    if global_rank == 0:
        correct_count = sum([result[0] for result in gathered_results])
        total_count = sum([result[1] for result in gathered_results])
        qa_data = []
        for result in gathered_results:
            qa_data.extend(result[2])

        print(f"Final Accuracy (all processes): {correct_count / total_count * 100:.2f}%")
        save_log_file(args.output_dir, args.output_name, correct_count, total_count, args)
        save_to_json(args.output_dir, f"{args.output_name}.json", qa_data)

    dist.barrier()
    dist.destroy_process_group()

def split_data(qa_data, num_processes, process_id):
    read_start = process_id * len(qa_data) // num_processes
    if process_id == num_processes - 1:
        read_end = len(qa_data)
    else:
        read_end = (process_id + 1) * len(qa_data) // num_processes
    qa_data = qa_data[read_start:read_end]

    return qa_data

def save_to_json(output_dir, output_name, data):
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def save_log_file(output_dir, output_name, correct_count, total_count, args):
    log_path = os.path.join(output_dir, f"{output_name}.log")
    if os.path.exists(output_dir):
        with open(log_path, 'a') as file:
            file.write("========================================\n")
            file.write(f"Final Accuracy (all processes): {correct_count / total_count * 100:.2f}%\n")
            file.write(f"Arguments: {args}\n")
    else:
        with open(log_path, 'w') as file:
            file.write(f"Final Accuracy (all processes): {correct_count / total_count * 100:.2f}%\n")
            file.write(f"Arguments: {args}\n")

if __name__ == "__main__":
    sys.path.append('../../')

    from llavidal.eval.model_utils import initialize_model, load_video
    from llavidal.inference import llavidal_infer as model_infer

    main()