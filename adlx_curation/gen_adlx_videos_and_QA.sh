#!/bin/bash

# Step 1: Generate person augmented videos from NTU dataset
python step1_person_augmented_generation.py --ntu_data_path /path/to/NTU120 --new_dataset_name NTU-PAG_224x224 --slack 64 --new_shape 224 224

# Step 2: Temporally stitch the videos. They are saved to save_dir
python step2_temporal_stitching.py --cropped_ntu_dir /path/to/NTU-PAG_224x224 --save_dir /directory/to/save/stitched/videos/ --video_mapping_json ./video_mapping.json

# Step 3: Generate frame level captions from stitched videos. Will generate "./cogvlm_{args.proc_num}.json" in the current directory. You need to combine these jsons into a single one.
## This script can be run in parallel to speed up the process. We use 8 GPUs and run the following command.
seq 1 7 | parallel --tag "python step3_image_frame_captions.py --num_procs 1 --stitched_vid_path /path/to/stitched/videos/ --num_procs 8 --proc_num {}"

# Step 4: Generate weakly supervised video descriptions from the action labels and the frame level captions. The output will be saved to "./dense_descriptions.json"
python step4_WS_video_desc.py --step3_description_json cogvlm-captions_FromSTEP3.json --openai_api_key openai_api_key

# Step 5: Generate the QA pairs that are used to train the model
python step5_generate_QA_pairs.py --step3_image_captions_path cogvlm-captions_FromSTEP3.json --step4_video_descriptions_path dense_descriptions.json --save_dir /directory/to/save/QA_pairs/ --openai_api_key openai_api_key