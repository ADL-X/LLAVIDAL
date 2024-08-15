# <img src="./llavidal/static/llavidal.ico"  style="vertical-align:middle;"/> LLAVIDAL 🏃👩‍🦯‍➡️🗨️

### LLAVIDAL: Benchmarking Large LAnguage VIsion Models for Daily Activities of Living

#### [Rajatsubhra Chakraborty](https://chakrabortyrajatsubhra.github.io)<sup>1</sup>* , [Arkaprava Sinha](https://webpages.charlotte.edu/asinha13/)<sup>1</sup>* , [Dominick Reilly](https://dominickrei.github.io)<sup>1</sup>*, [Manish Kumar Govind](https://manishgovind.github.io/)<sup>1</sup>, [Pu Wang](https://webpages.charlotte.edu/pwang13/)<sup>1</sup>, [Francois Bremond](http://www-sop.inria.fr/members/Francois.Bremond/)<sup>2</sup>, and [Srijan Das](https://srijandas07.github.io)<sup>1</sup>
\* Equally contributing first authors

##### Affiliations:
<sup>1</sup> University of North Carolina at Charlotte  
<sup>2</sup> Inria, Université Côte d’Azur

This codebase is adapted from [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT).

## News :loudspeaker:
- [Jun 13, 2024] [Paper](https://arxiv.org/pdf/2406.09390), [Instruction Set](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/EgbjE8ihBMBMjFyvYliaqQYBgRTqCEEgj8YH0JxJvl5nsQ?e=DxsZr6), [Evaluation Dataset](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/Eu2URrInCM5NuNwwGQqddrEBlxSOwuRyJkh1JvPuza-13g?e=Ec50Bc), and [Model Weights](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/EulRwM1VLaNLsm7rYxBMFfoBvUzw5nddl_4U9qSrWFZfIA?e=iXSflE) are released!
---

| Paper | Offline Demo | Training | Video Instruction Data | Quantitative Evaluation 
| :---: | :---: | :---: | :---: | :---: | 
| [![paper](https://img.shields.io/badge/Paper-<COLOR>.svg)](https://arxiv.org/pdf/2406.09390) | [Offline Demo](#Running-demo-🚗) | [Training](#Training-💪🦾) | [Video Instruction Dataset](https://studentuncc-my.sharepoint.com/personal/asinha13_charlotte_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fasinha13%5Fcharlotte%5Fedu%2FDocuments%2FLLAVIDAL%5Ffeatures%2Finstruction%5Fdata&ga=1) | [Quantitative Evaluation](https://studentuncc-my.sharepoint.com/personal/asinha13_charlotte_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fasinha13%5Fcharlotte%5Fedu%2FDocuments%2FLLAVIDAL%5Ffeatures%2Fevaluation&ga=1) |
## LLAVIDAL Overview 👁️:

LLAVIDAL (Large LAnguage VIsion model for Daily Activities of Living) is a multimodal model designed to understand and generate meaningful conversations about activities of daily living (ADL) performed by humans in videos. Its architecture integrates multiple modalities, including video, 3D human poses, and object interaction cues, with a large language model (LLM). Here's an overview of LLAVIDAL's Approach:

<p align="center">
  <img src="./llavidal/static/adlxteaser.gif" alt="LLAVIDAL Approach Overview">
</p>   



## Contributions ⭐

We introduce ADL-X, the first multiview RGBD instruction ADL dataset, curated through a
novel semi-automated framework for training LLVMs.

• LLAVIDAL is introduced as the first LLVM tailored for ADL, incorporating 3D poses and
object cues into the embedding space of the LLM.

• A new benchmark, ADLMCQ, is proposed for an objective evaluation of LLVMs on ADL
tasks, featuring MCQ tasks for action recognition & forecasting.

• Exhaustive experiments are conducted to determine the optimal strategy for integrating
poses or objects into LLAVIDAL. Evaluation of existing LLVMs on ADLMCQ and video
description tasks reveals that LLAVIDAL trained on ADL-X significantly outperforms
baseline LLVMs

---
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
To run the LLAVIDAL demo on your local GPU machine, please adhere to the following steps. Keep in mind that the demo requires around 18 GB of GPU memory.

1. Follow the installation instructions above
2. Download the LLAVIDAL weights from the following [link](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/EulRwM1VLaNLsm7rYxBMFfoBvUzw5nddl_4U9qSrWFZfIA?e=iXSflE)
3. Download LLaVa weights from this [link](https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1)

Finally, run the demo by executing the following command:
```shell
python llavidal/demo/video_demo.py \
    --model-name <path to the LLaVA-7B-Lightening-v1-1 weights downloaded in step 3> \
    --projection_path <path to the downloaded llavidal weights downloaded in step 2>
```

After running the command, follow the on-screen instructions to access the demo dashboard.
---

## Training 💪🦾

We train LLAVIDAL model on our 100K video instruction dataset. We initialize the training from LLaVA.
Please follow the instructions below to train LLAVIDAL-7B model.
Prepare LLaVA weights
LLAVIDAL is build using LLaVA. Please follow the following instructions of VideoChatGPT to get LLaVA weights.

Get the original LLaMA weights in the Hugging Face format.
Use the following scripts to get LLaVA weights by applying our delta.
```shell
python scripts/apply_delta.py \ 
        --base-model-path <path to LLaMA 7B weights> \
        --target-model-path LLaVA-Lightning-7B-v1-1 \
        --delta-path liuhaotian/LLaVA-Lightning-7B-delta-v1-1
```
The above command will download the LLaVA-Lightening-7B-v1-1 delta from Hugging Face, apply it to the provided LLaMA
weights and save the LLaVA-Lightening-7B-v1-1 weights in the current directory.
Alternatively you can download the ready LLaVA-Lightening-7B weights from [mmaaz60/LLaVA-Lightening-7B-v1-1](https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1)
Prepare Dataset
1. Download our [ADLX dataset](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/Ev529ctt_25ArNXTWqPF7NsBdM_m0A1SiIg9Qc1DKdCM2w?e=udCypF) video features.
   or
   Curate the dataset by following the steps in [[Video Instruction Dataset]].
2. Convert the downloaded [NTU_QA.json](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/EgbjE8ihBMBMjFyvYliaqQYBgRTqCEEgj8YH0JxJvl5nsQ?e=DxsZr6) into the required format for training,
```shell
python scripts/convert_instruction_json_to_training_format.py \
        --input_json_file <path to json file downloaded in step 2> \
        --output_json_file llavidal_training.json
The above script will generate llavidal_training.json required to train our model.
```
3. Prepare Spatio-Temporal features using CLIP
Note that for training efficiency, we pre-computed the video spatio-temporal features and use them directly during training.
After downloading the videos, please use the following command to generate CLIP spatio-temporal features.
 ```shell
 python scripts/save_spatio_temporal_clip_features.py \
        --video_dir_path <path to the directory containing all the videos> \
        --clip_feat_path <The output dir to save the features in.>
```
The script will generate the spatiotemporal features for each video and
save one pickle file per video in directory specified by --clip_feat_path argemunt.
Alternatively, you can download the pre-computed spatiotemporal CLIP features from [here](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/Ev529ctt_25ArNXTWqPF7NsBdM_m0A1SiIg9Qc1DKdCM2w?e=udCypF).

5. We are providing object features, pose features which are used as additional cues in the training. Which can be downloaded from here. We use the object features as our final model as it shows superior capabilities through our evaluation metrics.

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

The dataset is in [LINK](https://studentuncc-my.sharepoint.com/personal/asinha13_charlotte_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fasinha13%5Fcharlotte%5Fedu%2FDocuments%2FLLAVIDAL%5Ffeatures). The folders are [Video_features](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/Ev529ctt_25ArNXTWqPF7NsBdM_m0A1SiIg9Qc1DKdCM2w?e=udCypF) , [Pose Features](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/EuVbSocni-VAtHzfupCBmasBmBUaB9QGDEMWTHxBH_SApA?e=m0tDBP) and [Object Features](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/EnpaUynIshdDmKiPk5coSBABOVimC-dc46LHF55NOzpJ1g?e=0t646T)

If you want to recreate our dataset curation pipeline you can do so in the following steps:

Step 1: Download [NTURGBD dataset](https://rose1.ntu.edu.sg/dataset/actionRecognition/),follow the steps to get the dataset.

Step 2: Download the action combination list we created [ACTION LIST](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/Eo4hF_wpVGFMnMC5aEb-2LAB2rw1i3_XU2VV9BrOyeJDQg?e=LkIS4B).

Step 3: Arrange the NTU videos in Performer folders like P001,P002, etc

Step 4: Run the code, 
``` shell 
python /data_annotation/process_video_sequences.py
```
and pass the action combination list and video folder paths.

Step 5: Download and setup [CogVLM](https://github.com/THUDM/CogVLM). Follow the instructions to deploy the huggingface version to get frame-level annotations at 0.5fps. Run the command of the CogVLM demo,
```shell
python cli_demo_hf.py --from_pretrained THUDM/cogvlm-chat-hf --quant 4
```

Step 6: Get dense descriptions from GPT 3.5 Turbo using command,
``` shell 
python /data_annotation/generate_dense_descriptions.py
```
Pass the appropiate paths of the files and your OPENAI api key.

Step 6: Get QA pairs by running command,
``` shell 
python /data_annotation/generate_QA_pairs.py
```
Pass the previous made dense captions here and your OPENAI api key.

Alternatively you can access our **[TRAINING_DATA](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/EgbjE8ihBMBMjFyvYliaqQYBgRTqCEEgj8YH0JxJvl5nsQ?e=DxsZr6)** here if you want to skip the above process. We have provided both jsons, the final json that would be used for training is instruction_converted_training_data.json or else you can follow scripts to convert it yourself the NTU_QA.json to instruction data.

You can adapt the above process for your own ADL dataset curation with any ADL data just create your own action combinations like that of STEP 2.



**It is important to note we preprocessed our data to have person-centric cropping using Poses.**

**We highlight in our paper, why person-centric cropping is necessary for ADL Instruction Data Curation**

---

## Quantitative Evaluation 🧪

We introduce two new evaluation for ADL centric tasks -- [ADLMCQ-AR & ADLMCQ-AF](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/Eu2URrInCM5NuNwwGQqddrEBlxSOwuRyJkh1JvPuza-13g?e=Ec50Bc) which are MCQs conttaining Action Recognition and Action Forecasting Tasks.
We also release [SmartHome Untrimmed Descriptions](https://studentuncc-my.sharepoint.com/:f:/g/personal/asinha13_charlotte_edu/Eu2URrInCM5NuNwwGQqddrEBlxSOwuRyJkh1JvPuza-13g?e=Ec50Bc) for the first time.

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


