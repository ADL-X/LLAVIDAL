import argparse, datetime, random, glob, json, time, sys, os
from tqdm import tqdm
import numpy as np

import torch.distributed as dist
import torch

import openai

import warnings
warnings.filterwarnings("ignore")

import videochatgpt_scoring

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", help='Path to the LLaVA-7B-Lightening-v1-1 directory containing base weights of the model.', type=str, required=True)
    parser.add_argument("--proj_weight_path", help='Path to the .bin file containing projection weights of our model.', type=str, required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True, default='')
    parser.add_argument('--gt_file', help='Path to the json file containing video descriptions.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument('--max_visual_tokens', type=int, default=None, help='Maximum number of visual tokens.')
    parser.add_argument("--debug", action='store_true', help='Debug mode.')
    parser.add_argument("--seed", type=int, default=127, help='Random seed.')
    parser.add_argument("--openai_api_key", type=str, required=True, help='OpenAI API key for GPT-3.5 Turbo.')
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

    client = openai.OpenAI(api_key=args.openai_api_key)

    #### Start of run_inference code ####
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.base_model_path, args.proj_weight_path)
    conv_mode = 'llavidal_v1'

    gt_data = []
    with open(args.gt_file) as file:
        gt_data_raw = json.load(file)

    for vid_name, vid_desc in gt_data_raw.items():
        to_append = {}
        to_append['video_name'] = vid_name
        to_append['full_vid_desc'] = vid_desc
        to_append['subclip_paths'] = glob.glob(f'{args.video_dir}/{vid_name}_*.mp4')

        assert len(to_append['subclip_paths']) > 0, f"No subclip paths found for video '{vid_name}' in directory '{args.video_dir}'."

        gt_data.append(to_append)

    # This is just a naive split, number of clips per video differs so some GPUs will process more clips than others leading to hang time of some GPUs. If this becomes a problem, we can split based on clip duration to balance the load more evenly
    gt_data = split_data(gt_data, world_size, global_rank)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not args.debug or local_rank == 0:
        iterator = tqdm(enumerate(gt_data), total=len(gt_data))
    else:
        iterator = enumerate(gt_data)

    # metrics
    correctness_sum = 0
    detail_orientation_sum = 0
    contextual_sum = 0
    temporal_sum = 0
    consistency_sum = 0
    total_count = 0

    # hardcoded description question
    desc_question = "Please describe the primary actions and interactions in the video, focusing on movements and the use of objects by any person or persons present."

    for i, sample in iterator:
        total_count += 1

        gt_full_video_desc = sample['full_vid_desc']
        per_clip_general_descriptions = []

        # we first need to get the per-clip descriptions
        for clip_path in sample['subclip_paths']:
            if os.path.exists(clip_path):
                try:
                    frames = load_video(clip_path)
                except Exception as e:
                    # Handle the case where loading the video fails
                    print(f"(process {global_rank}) Failed to load video file '{clip_path}': {str(e)}")
                    continue
            else:
                print(f"Video file '{clip_path}' does not exist.")
                continue

            '''
            # > General descriptions
            '''
            try:
                prediction_general = model_infer(frames, desc_question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, max_new_tokens=args.max_new_tokens)

                per_clip_general_descriptions.append(prediction_general)

            except Exception as e:
                # raise e
                print(f"Error processing video file '{clip_path}': {str(e)}")
                continue

        mega_caption = ' '.join(per_clip_general_descriptions)
        full_vid_pred_summary = merge_clip_desc(client, mega_caption)
        # gt_data[i]['general_mega_caption'] = mega_caption
        gt_data[i]['general_pred_full_vid_summary'] = full_vid_pred_summary

        correctness = videochatgpt_scoring.get_correctness_score(desc_question, gt_full_video_desc, full_vid_pred_summary)
        detail_orientation = videochatgpt_scoring.get_detail_orientation_score(desc_question, gt_full_video_desc, full_vid_pred_summary)
        contextual = videochatgpt_scoring.get_context_score(desc_question, gt_full_video_desc, full_vid_pred_summary)
        temporal = videochatgpt_scoring.get_temporal_score(desc_question, gt_full_video_desc, full_vid_pred_summary)

        gt_data[i]['score_correctness'] = correctness
        gt_data[i]['score_detail_orientation'] = detail_orientation
        gt_data[i]['score_contextual'] = contextual
        gt_data[i]['score_temporal'] = temporal

        if args.debug and local_rank == 0:
            print(f"\n{'='*16} Question (General) {'='*16}")
            print(desc_question)
            print(f"\n{'='*16} Answer from LLM {'='*16}")
            print(full_vid_pred_summary)
            print(f"\n{'='*16} Scores & true ans {'='*16}")
            print(f'Correctness: {correctness}, Detail: {detail_orientation}, Contextual: {contextual}, Temporal: {temporal}\n')

        correctness_sum += correctness
        detail_orientation_sum += detail_orientation
        contextual_sum += contextual
        temporal_sum += temporal

        '''
        # > Consistency
        '''
        question_cons_1 = "Describe the actions in the scene"
        question_cons_2 = "What are the actions performed by the person in the video?"
        answer_cons = gt_full_video_desc

        cons_1_answers = []
        cons_2_answers = []

        for clip_path in sample['subclip_paths']:
            if os.path.exists(clip_path):
                try:
                    frames = load_video(clip_path)
                except Exception as e:
                    # Handle the case where loading the video fails
                    print(f"(process {global_rank}) Failed to load video file '{clip_path}': {str(e)}")
                    continue
            else:
                print(f"Video file '{clip_path}' does not exist.")
                continue

            try:
                prediction_cons_1 = model_infer(frames, question_cons_1, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, max_new_tokens=args.max_new_tokens)

                prediction_cons_2 = model_infer(frames, question_cons_2, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, max_new_tokens=args.max_new_tokens)

                cons_1_answers.append(prediction_cons_1)
                cons_2_answers.append(prediction_cons_2)

            except Exception as e:
                print(f"Error processing video file '{clip_path}': {str(e)}")
                continue

        # have answers for each subclip, now get a single answer for the entire video by merging the clip descriptions
        mega_caption_cons_1 = ' '.join(cons_1_answers)
        mega_caption_cons_2 = ' '.join(cons_2_answers)

        summarized_cons_1 = merge_clip_desc(client, mega_caption_cons_1)
        summarized_cons_2 = merge_clip_desc(client, mega_caption_cons_2)

        # gt_data[i]['cons_1_mega_caption'] = mega_caption_cons_1
        gt_data[i]['cons_1_pred_full_vid_summary'] = summarized_cons_1
        # gt_data[i]['cons_2_mega_caption'] = mega_caption_cons_2
        gt_data[i]['cons_2_pred_full_vid_summary'] = summarized_cons_2

        consistency = videochatgpt_scoring.get_consistency_score(question_cons_1, question_cons_2, answer_cons, summarized_cons_1, summarized_cons_2)

        gt_data[i]['score_consistency'] = consistency
        consistency_sum += consistency

        if not args.debug or local_rank == 0:
            iterator.set_description(f"{args.output_name} (process {local_rank}) {correctness_sum / total_count:.2f}/{detail_orientation_sum / total_count:.2f}/{contextual_sum / total_count:.2f}/{temporal_sum / total_count:.2f}/{consistency_sum / total_count:.2f}")
    #### End of run_inference code ####

    del model
    torch.cuda.empty_cache()

    results = [correctness_sum, detail_orientation_sum, contextual_sum, temporal_sum, consistency_sum, total_count, gt_data]

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

        gt_data = []
        for result in gathered_results:
            gt_data.extend(result[6])

        print(f'Final correctness (all processes): {correctness_final:.2f}')
        print(f'Final detail orientation (all processes): {detail_orientation_final:.2f}')
        print(f'Final contextual (all processes): {contextual_final:.2f}')
        print(f'Final temporal (all processes): {temporal_final:.2f}')
        print(f'Final consistency (all processes): {consistency_final:.2f}')
        print(f'Final average (all processes): {(correctness_final + detail_orientation_final + contextual_final + temporal_final + consistency_final) / 5:.2f}')

        save_log_file(args.output_dir, args.output_name, correctness_final, detail_orientation_final, contextual_final, temporal_final, consistency_final, args)
        save_to_json(args.output_dir, f"{args.output_name}.json", gt_data)

    dist.barrier()
    dist.destroy_process_group()

# specific for TSU descriptions, we have subclips so we need to merge the clip descriptions into a final summary
def merge_clip_desc(client, mega_caption):
    try:
        completion = client.chat.completions.create(model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "system",
                    "content": ("You are an intelligent chatbot designed for video description and summarization. "
                                "Your task is to generate a detailed and cohesive summary based on the provided mega caption of a video. "
                                "------"
                                "##TASK:"
                                "Users will provide a mega caption of a video, and you will generate a detailed summary. "
                                "The summary should be a cohesive and well-structured paragraph based on the provided mega caption, "
                                "capturing all the actions and objects in approximately 300 words. "
                                "------"
                                "##INSTRUCTIONS:"
                                "- Generate a comprehensive summary of approximately 300 words."
                                "- The summary must be a cohesive and detailed version of the provided mega caption."
                                "- Combine the information from the mega caption into a single coherent summary, ignoring any repetitions."
                                "- Give more emphasis on the actions, the objects, and the colors of the background and the objects."
                                "- Describe the sequence of actions happening in the video and the objects the person interacts with.")
                },
                {
                    "role": "user",
                    "content": f"The mega caption of the video is: {mega_caption}. "
                            "Please generate a detailed 300-word summary that describes the video content, "
                            "focusing on actions, objects, and the environment."
                }
            ],
            temperature=0.7,
            max_tokens=500  # Adjust this value based on your needs
        )

        try:
            summary = completion.choices[0].message.content
            return summary
        except Exception as e:
            print(f"Error processing response: {e}. Skipping this item.")
            return {"error": "Failed to process response."}
    except openai.RateLimitError as e:
        print(f"Rate limit exceeded: {e}. Waiting before retrying...")
        time.sleep(60)  # Wait for 60 seconds before retrying
        return merge_clip_desc(mega_caption)  # Retry the request
    except Exception as e:
        print(f"An error occurred: {e}. Skipping this item.")
        return {"error": "An unknown error occurred."}   

def split_data(gt_data, num_processes, process_id):
    read_start = process_id * len(gt_data) // num_processes
    if process_id == num_processes - 1:
        read_end = len(gt_data)
    else:
        read_end = (process_id + 1) * len(gt_data) // num_processes
    gt_data = gt_data[read_start:read_end]

    return gt_data

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