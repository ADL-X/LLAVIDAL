## Paths to video directories and QA files
CHARADES_VID_PATH="/path/to/Charades_videos"
LEMMA_VID_PATH="/path/to/LEMMA_videos"
SMARTHOME_VID_PATH="/path/to/Smarthome_videos"
TSU_TC_VID_PATH="/path/to/TSU_TC_videos"
TSU_DESC_VID_PATH="/path/to/TSU_Descriptions_videos"

CHARADES_AR="/path/to/Charades-AR.json"
LEMMA_TC="/path/to/LEMMA-TC.json"
SMARTHOME_AR="/path/to/Smarthome-AR.json"
TSU_TC="/path/to/TSU-TC.json"

CHARADES_DESCRIPTIONS="/path/to/Charades-Description.json"
TSU_DESCRIPTIONS="/path/to/TSU-Description.json"

BASE_LLAVA_PATH="/path/to/LLaVA-7B-Lightening-v1-1/"

OUTPUT_DIR="/path/to/output/directory" # change to your desired output directory
DEBUG="--debug" # set to "" to disable, "--debug" to enable

NUM_GPUS=$1 # number of GPUs to use for distributed evaluation



# list of model paths to evaluate
MODEL_PATHS=(
    "/path/to/your/model1"
    # "/path/to/your/model2"
)

# list of corresponding output names
OUTPUT_NAMES=(
    "model1"
    # "model2"
)

for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATHS[$i]}
    OUTPUT_NAME=${OUTPUT_NAMES[$i]}

    # Charades-AR, Smarthome-AR, LEMMA-TC, TSU-TC evaluations
    torchrun --nproc_per_node=$NUM_GPUS eval_adlxmcq.py --base_model_path $BASE_LLAVA_PATH --proj_weight_path $MODEL_PATH --video_dir $CHARADES_VID_PATH --qa_file $CHARADES_AR --output_dir $OUTPUT_DIR --output_name Charades-AR-$OUTPUT_NAME $DEBUG
    torchrun --nproc_per_node=$NUM_GPUS eval_adlxmcq.py --base_model_path $BASE_LLAVA_PATH --proj_weight_path $MODEL_PATH --video_dir $SMARTHOME_VID_PATH --qa_file $SMARTHOME_AR --output_dir $OUTPUT_DIR --output_name Smarthome-AR-$OUTPUT_NAME $DEBUG
    torchrun --nproc_per_node=$NUM_GPUS eval_adlxmcq.py --base_model_path $BASE_LLAVA_PATH --proj_weight_path $MODEL_PATH --video_dir $LEMMA_VID_PATH --qa_file $LEMMA_TC --output_dir $OUTPUT_DIR --output_name LEMMA-TC-$OUTPUT_NAME $DEBUG
    torchrun --nproc_per_node=$NUM_GPUS eval_adlxmcq.py --base_model_path $BASE_LLAVA_PATH --proj_weight_path $MODEL_PATH --video_dir $TSU_TC_VID_PATH --qa_file $TSU_TC --output_dir $OUTPUT_DIR --output_name TSU-TC-$OUTPUT_NAME $DEBUG

    # Charades-Descriptions
    torchrun --nproc_per_node=$NUM_GPUS eval_charades_desc.py --base_model_path $BASE_LLAVA_PATH --proj_weight_path $MODEL_PATH --video_dir $CHARADES_VID_PATH --qa_file $CHARADES_DESCRIPTIONS --output_dir $OUTPUT_DIR --output_name Charades-Desc-GenCons-$OUTPUT_NAME --proj_weight_path $MODEL_PATH $DEBUG

    # TSU-Descriptions
    # !!!!! requires OpenAI API key to be exported (e.g., in ~/.bashrc) !!!!!
    torchrun --nproc_per_node=$NUM_GPUS eval_tsu_desc.py --base_model_path $BASE_LLAVA_PATH --proj_weight_path $MODEL_PATH --video_dir $TSU_DESC_VID_PATH --gt_file $TSU_DESCRIPTIONS --output_dir $OUTPUT_DIR --output_name TSU-Desc-GenCons-$OUTPUT_NAME --proj_weight_path $MODEL_PATH $DEBUG --openai_api_key $OPENAI_API_KEY
done