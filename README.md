# <img src="./llavidal/static/llavidal.ico"  style="vertical-align:middle;"/> LLAVIDAL: Benchmarking Large LAnguage VIsion Models for Daily Activities of Living 🏃👩‍🦯‍➡️🗨️

<p align="center">
  <img src="./llavidal/static/adlxteaser.gif" alt="LLAVIDAL Approach Overview">
</p>   

This codebase is adapted from [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT).

-----
## Available resources
| **Resource**               | **Link**                                                                                                                                                                                            |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper**                  | [![Paper](https://img.shields.io/badge/Read-Paper-blue.svg)](https://arxiv.org/pdf/2406.09390)                                                                                                      |
| **LLAVIDAL Weights**          | [![Weights](https://img.shields.io/badge/Download-Model_Weights-green.svg)](https://huggingface.co/datasets/dreilly/ADL-X/tree/main/model_weights)     |
| **(ADL-X) Multi-modal Features**         | [![Multi-modal Features](https://img.shields.io/badge/Download-Multimodal_Features-orange.svg)](https://huggingface.co/datasets/dreilly/ADL-X/tree/main/multimodal_features) |
| **(ADL-X) Instruction Dataset**    | [![Instruction Dataset](https://img.shields.io/badge/Download-Instruction_Dataset-yellowgreen.svg)](https://huggingface.co/datasets/dreilly/ADL-X/tree/main/instruction_data) |
| **Data Curation**          | [![Data Curation](https://img.shields.io/badge/Read-Data_Curation-aquamarine.svg)](#data-curation-pipeline-) |
| **Training**               | [![Training](https://img.shields.io/badge/Start-Training-crimson.svg)](#training-) |
| **Offline Demo**           | [![Offline Demo](https://img.shields.io/badge/Run-Offline_Demo-teal.svg)](#running-demo-) |
| **Quantitative Evaluation**| [![Quantitative Evaluation](https://img.shields.io/badge/View-Quantitative_Evaluation-lightgrey.svg)](#quantitative-evaluation-) |

---

## What is your goal?
<table>
    <thead><tr><th colspan=4><a href="#installation-wrench">Start with installation (click here)</a></th></tr></thead>
    <tbody><tr>
        <td><a href="#running-demo-">I want to try the demo</td>
        <td>I want to train LLAVIDAL</td>
        <td>I want to reproduce LLAVIDAL results</td>
        <td>I want to generate the ADL-X dataset</td>
    </tr></tbody>
</table>

## LLAVIDAL Architecture ⚙️

<p align="center">
  <img src="./llavidal/static/ADL-architecture.png" alt="LLAVIDAL Architecture Overview">
</p>   


Overview of LLAVIDAL, which utilizes an LLM to integrate multiple modalities, including
video, pose, and object features. Videos are represented by embeddings obtained from a VLM, poses
are processed through (PoseLM), and object embeddings are obtained through (ObjectLM). These
embeddings are projected into the LLM space, where they are concatenated with tokenized text
queries for instruction tuning.

---

## Installation :wrench:
Our python environement is identical to [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT), we recommend following their installation instructions:

```shell
conda create --name=llavidal python=3.10
conda activate llavidal

git clone https://github.com/ADL-X/LLAVIDAL.git
cd LLAVIDAL
pip install -r requirements.txt

export PYTHONPATH="./:$PYTHONPATH"
```

Additionally, if you are using A100/H100 GPUs you can install [FlashAttention](https://github.com/HazyResearch/flash-attention),
```shell
pip install ninja

git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
git checkout v1.0.7
python setup.py install
```
---

## Running Demo 🚗
We provide a Gradio demo to run LLAVIDAL on your local machine. For the best performance, use a CUDA-enabled machine with at least 18GB of VRAM.

1. Activate the llavidal Conda environment `conda activate llavidal`
2. Download the following LLaVA weights: [LLaVA-7B-Lightening-v1-1](https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1)
3. Download the LLAVIDAL weights from [Available Resources](#available-resources)

Finally, run the demo by executing the following command:

```shell
python llavidal/demo/video_demo.py \
    --model-name <path to the LLaVA-7B-Lightening-v1-1 weights downloaded in step 2> \
    --projection_path <path to the downloaded llavidal weights (llavidal_weights.bin
) downloaded in step 3>
```

After running the command a URL will be provided. Click this URL and follow the on-screen instructions to use the demo.

---

## Training 💪🦾

LLAVIDAL is trained on ADL-X, an ADL dataset of over 100K video-instruction pairs. The weights of the model are initialized from LLaVA and is trained for 3 epochs on 8 48GB NVIDIA RTX A6000 GPUs. To begin, download the LLaVA weights from this link: [LLaVA-7B-Lightening-v1-1](https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1).

We provide two methods to prepare the ADL-X dataset:
1. **Download the pre-extracted features (quickest)**:
   - Download the Multi-modal Features (`video_features.zip`, `object_features.zip`, `pose_features.zip`) and Instruction Dataset (`NTU_QA-for-training.json`) from [Available Resources](#available-resources)
2. **Curate the dataset and extract the features**:
   - Follow the steps in the [Data Curation Pipeline](#data-curation-pipeline-).

6. Train LLAVIDAL
We have trained on 8 A6000 40GB GPUs using the following command,
```shell
torchrun --nproc_per_node=8 --master_port 29001 llavidal/train/train_mem.py \
          --model_name_or_path <path to LLaVA-7B-Lightening-v-1-1 model> \
          --version v1 \
          --data_path <path to the llavidal using `convert_instruction_json_to_training_format.py` script.> \
          --video_folder <path to the spatio-temporal features generated in step 4 using `save_spatio_temporal_clip_features.py` script> \
          --object_folder <path to the downloaded object features>/
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./LLAVIDAL_7B-1.1_Checkpoints \
          --num_train_epochs 3 \
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 3000 \
          --save_total_limit 3 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 True \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True

```
 You can change the object features to pose features and change one line in the code to pass train_pose.py and llavidal_pose.py in train_mem.py. Similarly, for both object and pose features use train_pose_object.py and llavidal_pose_object.py. Pass the object and pose path together in that case.


---

## Video Instruction Dataset :📂

We are introducing ADLX the first ADL centric video instruction dataset, due to licensing restrictions we cannot share the original videos but we are providing the video features of our dataset,we are also providing the object features and the pose features.

The dataset is in [LINK](https://huggingface.co/datasets/dreilly/ADL-X/tree/main). The folders are [Video_features](https://huggingface.co/datasets/dreilly/ADL-X/blob/main/multimodal_features/object_features.zip) , [Pose Features](https://huggingface.co/datasets/dreilly/ADL-X/blob/main/multimodal_features/pose_features.zip) and [Object Features](https://huggingface.co/datasets/dreilly/ADL-X/blob/main/multimodal_features/object_features.zip)

The video features are like 
```
Video Features
├── 001_video_001.pkl
├── 001_video_002.pkl
├── 001_video_003.pkl
├── 001_video_004.pkl
├── 001_video_005.pkl
├── 001_video_006.pkl
...

```
each video feature is of dimension 356 x 1024.

The pose features are like
```
Pose Features
├── 001_001_video_001_pose.pickle
├── 001_001_video_002_pose.pickle
├── 001_001_video_003_pose.pickle
├── 001_001_video_004_pose.pickle
├── 001_001_video_005_pose.pickle
├── 001_001_video_006_pose.pickle
├── 001_001_video_007_pose.pickle
├── 001_001_video_008_pose.pickle
...
```
each pose feature is of the dimension 216 x 256

The object features are like
```
Object Features
├── 001_001_video_001_object.pkl
├── 001_001_video_002_object.pkl
├── 001_001_video_003_object.pkl
├── 001_001_video_004_object.pkl
├── 001_001_video_005_object.pkl
├── 001_001_video_006_object.pkl
├── 001_001_video_007_object.pkl
├── 001_001_video_008_object.pkl
...
```
each object feature is of the dimension n x 8x 512, where n is the number of objects present in the video.

## ADL-X Data Curation Pipeline 📖 

**NOTE: You can skip this process entirely and download the ADL-X dataset in the [Available Resources](#available-resources) section above**

Follow the steps below to recreate ADL-X using the data curation pipeline proposed in the paper.

**Step 1**: Download the [NTU-RGB+D dataset](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

**Step 2**: Download the action combination [here](https://huggingface.co/datasets/dreilly/ADL-X/blob/main/data_curation/all_action_combinations.txt).

**Step 3**: Arrange the directory structure of the NTU-RGB+D videos in the following way:

```
NTU Videos
├── P001
│   ├── S001C001P001R001A001_rgb.avi
│   ├── S001C001P001R001A002_rgb.avi
│   ├── S001C001P001R001A003_rgb.avi
│   ...
├── P015
│   ├── S003C001P015R001A001_rgb.avi
│   ├── S003C001P015R001A002_rgb.avi
│   ├── S003C001P015R001A003_rgb.avi
│   ├── S003C001P015R001A004_rgb.avi
│   ├── S003C001P015R001A005_rgb.avi
│   ...

...
```

**Step 4**: Generate the temporally stitched videos with the following command, passing the action combination list and video folder path.
``` shell 
python /data_annotation/process_video_sequences.py
```

**Step 5**: Download and install [CogVLM](https://github.com/THUDM/CogVLM). Run the following command (which uses [cli_demo_hf.py](https://github.com/THUDM/CogVLM/blob/main/basic_demo/cli_demo_hf.py)) to get frame-level annotations of the temporally stitched videos at 0.5fps:
```shell
python cli_demo_hf.py --from_pretrained THUDM/cogvlm-chat-hf --quant 4
```

**Step 6**: Obtain dense video descriptions using GPT 3.5 Turbo with the following command, passing the frame-level annotations from Step 5 and your OpenAI API key:
``` shell 
python data_annotation/generate_dense_descriptions.py
```

**Step 7**: Generate QA pairs by running the following command, passing the dense video descriptions from Step 6 and your OpenAI API key:
``` shell 
python /data_annotation/generate_QA_pairs.py
```

**Step 8**: Convert the generated QA pairs from Step 7 (also availble at [NTU_QA.json](https://huggingface.co/datasets/dreilly/ADL-X/tree/main/evaluation)) into the required format for training:
```shell
python scripts/convert_instruction_json_to_training_format.py \
        --input_json_file <path to json file downloaded in step 2> \
        --output_json_file NTU_QA-for-training.json
```

**Step 9**: Prepare Spatio-Temporal features using CLIP
For training efficiency, we pre-compute the spatio-temporal video features used during training. The following command will save one pickle file per video in directory specified by the `--clip_feat_path` argument. Run the following command to generate spatio-temporal features with CLIP:
 ```shell
 python scripts/save_spatio_temporal_clip_features.py \
        --video_dir_path <path to the directory containing videos generated in Step 4> \
        --clip_feat_path <The output dir to save the features in>
```

**Step 10**: Download the pre-computed pose (`pose_features.zip`) and object features (`object_features.zip`) from [Available Resources](#available-resources) and extract them

Our data curation pipeline can be adapted to trimmed (start at Step 2) or untrimmed (start at Step 5) video datasets.

---

## Quantitative Evaluation 🧪

We introduce two new evaluation for ADL centric tasks -- [ADLMCQ-AR & ADLMCQ-AF](https://huggingface.co/datasets/dreilly/ADL-X/tree/main/evaluation) which are MCQs conttaining Action Recognition and Action Forecasting Tasks.
We also release [SmartHome Untrimmed Descriptions](https://huggingface.co/datasets/dreilly/ADL-X/blob/main/evaluation/Video_description_Smarthome_Untrimmed.json) for the first time.

Step 1: Download all the datasets-- [Charades](https://prior.allenai.org/projects/charades) , [LEMMA](https://sites.google.com/view/lemma-activity)(We use the exo-view) ,[SMARTHOME UNTRIMMED and TRIMMED](https://project.inria.fr/toyotasmarthome/).

Step 2: For Action Forecasting access the json files and slice the videos from the start frame and end frame.For action recognition nothing is needed.


Step 3: Arrange the data like that in the json file provided and run the command ,
```shell
cd llavidal/eval/
```
```shell
python run_inference_action_recognition_charades.py
--video_dir /path/to/videos \
  --qa_file /path/to/qa_file.json \
  --output_dir /path/to/output \
  --output_name results \
  --model-name <LLAVA model path> \
  --conv-mode llavidal_v1 \
  --projection_path <path to LLAVIDAL WEIGHTS> 
```


Step 3: Evaulate using GPT3.5 Turbo api 
```shell
cd quantitative_evaluation/
```
```shell
evaluate_action_recognition_charades.py
```
and pass the above results in STEP 2.

For other methods the above steps are same 

-----------------
For video descriptions for Charades run command 

```shell
cd llavidal/eval
```
```shell
python run_inference_benchmark_general.py
```
Pass the appropiate paths to get the results josn

For video descriptions for Smarthome Untrimmed ,slice the videos in 1 minutes each and make a dense description like that of data curation process.

To get individual descriptions 

```shell
cd llavidal/eval
```
```shell
python run_inference_descriptions_smarthome.py
```


We closely follow the [MEMENTOS EVALUATION](https://github.com/si0wang/Mementos) to get the object and action F1 scores

We provide a notebook to achieve the execute the above approach.
Follow this notebook to get the evaluation 
 ```shell
cd quantitative_evaluation/mementos_evaluation.ipynb
```



---

## Qualitative Analysis 🎬

<p align="center">
  <img src="./llavidal/static/QA_example.png" alt="Qualitative Evaluation">
</p>  

---

## Acknowledgements 🙏

+ [LLaMA](https://github.com/facebookresearch/llama): Great step towards bridging vision and language!
+ [VideoChatgpt](https://github.com/mbzuai-oryx/Video-ChatGPT?tab=readme-ov-file): We thank for the foundational work.
+ [LLAVA](https://llava-vl.github.io) : For inspiring the overall architecture
+ [CogVLM](https://github.com/THUDM/CogVLM): For creating a strong captioning model.

If you're using LLAVIDAL in your research or applications, please cite using this BibTeX:
```bibtex
@misc{chakraborty2024llavidal,
      title={LLAVIDAL: Benchmarking Large Language Vision Models for Daily Activities of Living}, 
      author={Rajatsubhra Chakraborty and Arkaprava Sinha and Dominick Reilly and Manish Kumar Govind and Pu Wang and Francois Bremond and Srijan Das},
      year={2024},
      eprint={2406.09390},
      archivePrefix={arXiv},
      primaryClass={id='cs.CV' full_name='Computer Vision and Pattern Recognition' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'}
}
```
----------

## Usage LICENSE :

The dataset is protected under the CC-BY license of Creative Commons, which allows users to distribute, remix, adapt, and build upon the material in any medium or format, as long as the creator is attributed. The license allows ADL-X for commercial use. As the authors of this manuscript and collectors of this dataset, we reserve the right to distribute the data.

------


