import argparse, datetime, random, json, sys, os
from tqdm import tqdm
import numpy as np

import torch.distributed as dist
import torch

import warnings
warnings.filterwarnings("ignore")

import videochatgpt_scoring

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

    if not args.debug or local_rank == 0:
        iterator = tqdm(enumerate(qa_data), total=len(qa_data))
    else:
        iterator = enumerate(qa_data)

    # metrics
    correctness_sum = 0
    detail_orientation_sum = 0
    contextual_sum = 0
    temporal_sum = 0
    consistency_sum = 0
    total_count = 0

    for i, sample in iterator:
        total_count += 1

        video_path = os.path.join(args.video_dir, sample['video_name'] + '.mp4')
        
        if not os.path.exists(video_path):
            print(f"Video file '{video_path}' does not exist.")
            continue

        video_frames = load_video(video_path)
        
        '''
        # > General descriptions
        '''
        question_desc = sample['Q']
        answer_desc = sample['A']

        try:
            prediction_general = model_infer(video_frames, question_desc, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, max_new_tokens=args.max_new_tokens)

            qa_data[i]['prediction_general'] = prediction_general

            correctness = videochatgpt_scoring.get_correctness_score(question_desc, answer_desc, prediction_general)
            detail_orientation = videochatgpt_scoring.get_detail_orientation_score(question_desc, answer_desc, prediction_general)
            contextual = videochatgpt_scoring.get_context_score(question_desc, answer_desc, prediction_general)
            temporal = videochatgpt_scoring.get_temporal_score(question_desc, answer_desc, prediction_general)

            qa_data[i]['score_correctness'] = correctness
            qa_data[i]['score_detail_orientation'] = detail_orientation
            qa_data[i]['score_contextual'] = contextual
            qa_data[i]['score_temporal'] = temporal

            if args.debug and local_rank == 0:
                print(f"\n{'='*16} Question (General) {'='*16}")
                print(question_desc)
                print(f"\n{'='*16} Answer from LLM {'='*16}")
                print(prediction_general)
                print(f"\n{'='*16} Scores & true ans {'='*16}")
                print(f'Correctness: {correctness}, Detail: {detail_orientation}, Contextual: {contextual}, Temporal: {temporal}\n')

            correctness_sum += correctness
            detail_orientation_sum += detail_orientation
            contextual_sum += contextual
            temporal_sum += temporal

        except Exception as e:
            raise e
            print(f"Error processing video file '{video_path}': {str(e)}")

        '''
        # > Consistency
        '''
        question_cons_1 = sample['cons_Q1']
        question_cons_2 = sample['cons_Q2']
        answer_cons = sample['cons_A']

        try:
            prediction_cons_1 = model_infer(video_frames, question_cons_1, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, max_new_tokens=args.max_new_tokens)

            prediction_cons_2 = model_infer(video_frames, question_cons_2, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, max_new_tokens=args.max_new_tokens)

            qa_data[i]['prediction_cons_1'] = prediction_cons_1
            qa_data[i]['prediction_cons_2'] = prediction_cons_2

            consistency = videochatgpt_scoring.get_consistency_score(question_cons_1, question_cons_2, answer_cons, prediction_cons_1, prediction_cons_2)

            qa_data[i]['score_consistency'] = consistency

            consistency_sum += consistency

        except Exception as e:
            raise e

        if not args.debug or local_rank == 0:
            iterator.set_description(f"{args.output_name} (process {local_rank}) {correctness_sum / total_count:.2f}/{detail_orientation_sum / total_count:.2f}/{contextual_sum / total_count:.2f}/{temporal_sum / total_count:.2f}/{consistency_sum / total_count:.2f}")
    #### End of run_inference code ####

    del model
    torch.cuda.empty_cache()

    results = [correctness_sum, detail_orientation_sum, contextual_sum, temporal_sum, consistency_sum, total_count, qa_data]

    gathered_results = [None for _ in range(dist.get_world_size())]
    dist.gather_object(
        obj=results,
        object_gather_list=gathered_results if global_rank == 0 else None,
        dst=0,
    )

    if global_rank == 0:
        global_total_count = sum([result[5] for result in gathered_results])
        correctness_final = (sum([result[0] for result in gathered_results]) / global_total_count) * 20
        detail_orientation_final = (sum([result[1] for result in gathered_results]) / global_total_count) * 20
        contextual_final = (sum([result[2] for result in gathered_results]) / global_total_count) * 20
        temporal_final = (sum([result[3] for result in gathered_results]) / global_total_count) * 20
        consistency_final = (sum([result[4] for result in gathered_results]) / global_total_count) * 20

        qa_data = []
        for result in gathered_results:
            qa_data.extend(result[6])
            
        print(f'Final correctness (all processes): {correctness_final:.2f}')
        print(f'Final detail orientation (all processes): {detail_orientation_final:.2f}')
        print(f'Final contextual (all processes): {contextual_final:.2f}')
        print(f'Final temporal (all processes): {temporal_final:.2f}')
        print(f'Final consistency (all processes): {consistency_final:.2f}')
        print(f'Final average (all processes): {(correctness_final + detail_orientation_final + contextual_final + temporal_final + consistency_final) / 5:.2f}')

        save_log_file(args.output_dir, args.output_name, correctness_final, detail_orientation_final, contextual_final, temporal_final, consistency_final, args)
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

def save_log_file(output_dir, output_name, corr_final, do_final, context_final, temp_final, cons_final, args):
    log_path = os.path.join(output_dir, f"{output_name}.log")
    if os.path.exists(output_dir):
        with open(log_path, 'a') as file:
            file.write("========================================\n")
            file.write(f"Final correctness (all processes): {corr_final:.2f}%\n")
            file.write(f"Final detail orientation (all processes): {do_final:.2f}%\n")
            file.write(f"Final contextual (all processes): {context_final:.2f}%\n")
            file.write(f"Final temporal (all processes): {temp_final:.2f}%\n")
            file.write(f"Final consistency (all processes): {cons_final:.2f}%\n")
            file.write(f"Final average (all processes): {(corr_final + do_final + context_final + temp_final + cons_final) / 5:.2f}%\n")
            file.write(f"Arguments: {args}\n")
    else:
        with open(log_path, 'w') as file:
            file.write(f"Final correctness (all processes): {corr_final:.2f}%\n")
            file.write(f"Final detail orientation (all processes): {do_final:.2f}%\n")
            file.write(f"Final contextual (all processes): {context_final:.2f}%\n")
            file.write(f"Final temporal (all processes): {temp_final:.2f}%\n")
            file.write(f"Final consistency (all processes): {cons_final:.2f}%\n")
            file.write(f"Final average (all processes): {(corr_final + do_final + context_final + temp_final + cons_final) / 5:.2f}%\n")
            file.write(f"Arguments: {args}\n")

if __name__ == "__main__":
    sys.path.append('../../')

    from llavidal.eval.model_utils import initialize_model, load_video
    from llavidal.inference import llavidal_infer as model_infer

    main()