# <img src="./llavidal/static/llavidal.ico"  style="vertical-align:middle;"/> LLAVIDAL 🏃👩‍🦯‍➡️🗨️

### LLAVIDAL: Benchmarking Large LAnguage VIsion Models for Daily Activities of Living

#### [Rajatsubhra Chakraborty](https://chakrabortyrajatsubhra.github.io)<sup>1</sup>* , [Arkaprava Sinha](https://www.linkedin.com/in/arkaprava-sinha)<sup>1</sup>* , [Dominick Reilly](https://dominick-reilly.github.io/)<sup>1</sup>* , [Manish Kumar Govind](https://sites.google.com/view/manishkumargovind/home)<sup>1</sup>, [Pu Wang](https://webpages.charlotte.edu/pwang13/)<sup>1</sup>,[Francois Bremond](http://www-sop.inria.fr/members/Francois.Bremond/)<sup>2</sup> and [Srijan Das](https://srijandas07.github.io)<sup>1</sup>
\* Equally contributing first authors

##### Affiliations:
<sup>1</sup> University of North Carolina at Charlotte  
<sup>2</sup> INRIA, Université Côte d’Azur


## :loudspeaker: Latest Updates
Placeholder for updates

---


## LLAVIDAL Overview 👁️:

LLAVIDAL (Large LAnguage VIsion model for Daily Activities of Living) is a multimodal model designed to understand and generate meaningful conversations about activities of daily living (ADL) performed by humans in videos. Its architecture integrates multiple modalities, including video, 3D human poses, and object interaction cues, with a large language model (LLM). Here's an overview of LLAVIDAL's Approach:

<p align="center">
  <img src="./llavidal/static/adlxteaser.gif" alt="LLAVIDAL Approach Overview">
</p>   



## Contributions ⭐:

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
## LLAVIDAL Architecture:⚙️:

<p align="center">
  <img src="./llavidal/static/ADL-architecture.png" alt="LLAVIDAL Architecture Overview">
</p>   

---


## Installation :wrench:

We recommend setting up a conda environment for the project:
```shell
conda env create -f environment.yml

```
Additionally,if you have A100 you can  install [FlashAttention](https://github.com/HazyResearch/flash-attention) for training,
```shell
pip install ninja

git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
git checkout v1.0.7
python setup.py install
```


---

## Running Demo :🚗:

To run the LLAVIDAL demo on your local GPU machine, please adhere to the following steps. Keep in mind that the demo requires around 18 GB of GPU memory.

Clone the LLAVIDAL Repository

First, clone the LLAVIDAL repository by running the following commands in your terminal:
```shell 
git clone https://github.com/ADL-X/LLAVIDAL.git
```
cd llavidal
```shell
export PYTHONPATH="./:$PYTHONPATH"
```
Download LLAVIDAL Weights

Next, download the LLAVIDAL weights from this link:

Prepare LLaVA Weights

Since LLAVIDAL is built using LLaVA, you need to obtain the LLaVA weights by following these steps:

Obtain the original LLaMA weights in the HuggingFace format by referring to the instructions here.

Apply the LLaVA delta to the LLaMA weights using the provided script:

```shell
python scripts/apply_delta.py \
    --base-model-path <path to LLaMA 7B weights> \
    --target-model-path LLaVA-Lightning-7B-v1-1 \
    --delta-path liuhaotian/LLaVA-Lightning-7B-delta-v1-1
This command will download the LLaVA-Lightening-7B-v1-1 delta from HuggingFace, apply it to the specified LLaMA weights, and save the resulting LLaVA-Lightening-7B-v1-1 weights in the current directory.
```

Finally, run the demo by executing the following command:
```shell
python llavidal/demo/video_demo.py \
    --model-name <path to the LLaVA-Lightening-7B-v1-1 weights prepared in step 3> \
    --projection_path <path to the downloaded llavidal weights>
```
After running the command, follow the on-screen instructions to access the demo dashboard.

---

## Training :💪🦾:

We train LLAVIDAL model on our 100K video instruction dataset. We initialize the training from LLaVA.
Please follow the instructions below to train LLAVIDAL-7B model.
Prepare LLaVA weights
LLAVIDAL is build using LLaVA. Please follow the following instructions to get LLaVA weights.

Get the original LLaMA weights in the Hugging Face format by following the instructions here.
Use the following scripts to get LLaVA weights by applying our delta.
```shell
python scripts/apply_delta.py \ 
        --base-model-path <path to LLaMA 7B weights> \
        --target-model-path LLaVA-Lightning-7B-v1-1 \
        --delta-path liuhaotian/LLaVA-Lightning-7B-delta-v1-1
```
The above command will download the LLaVA-Lightening-7B-v1-1 delta from Hugging Face, apply it to the provided LLaMA
weights and save the LLaVA-Lightening-7B-v1-1 weights in the current directory.
Alternatively you can download the ready LLaVA-Lightening-7B weights from mmaaz60/LLaVA-Lightening-7B-v1-1.
Prepare Dataset
1. Download our 100K video instruction dataset from
this download link.
2. Convert the downloaded Json into the required format for training,
```shell
python scripts/convert_instruction_json_to_training_format.py \
        --input_json_file <path to json file downloaded in step 2> \
        --output_json_file llavidal_training.json
The above script will generate llavidal_training.json required to train our model.
```
3. Download NTU120RGBD videos
All the videos annotated in our work are taken from NTU120RGBD dataset.
We provide the ids of all the required videos in the train_video_ids.txt file.
Please follow the instructions on the official site to download the videos.
Alternatively, you can download these from here.
4. Prepare Spatio-Temporal features using CLIP
Note that for training efficiency, we pre-computed the video spatio-temporal features and use them directly during training.
After downloading the videos, please use the following command to generate CLIP spatio-temporal features.
 ```shell
 python scripts/save_spatio_temporal_clip_features.py \
        --video_dir_path <path to the directory containing all the videos> \
        --clip_feat_path <The output dir to save the features in.>
```
The script will generate the spatiotemporal features for each video and
save one pickle file per video in directory specified by --clip_feat_path argemunt.
Alternatively, you can download the pre-computed spatiotemporal CLIP features from here.

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

---

## Video Instruction Dataset :📂

We are introducing ADLX the first ADL centric video instruction dataset, due to licensing restrictions we cannot share the original videos but we are providing the video features of our dataset,we are also providing the object features and the pose features.

The dataset is in [LINK](https://studentuncc-my.sharepoint.com/personal/asinha13_uncc_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fasinha13%5Funcc%5Fedu%2FDocuments%2FLLAVIDAL%5Ffeatures&ga=1). The folders are [Video_features]( https://tinyurl.com/video-features) , [Pose Features]( https://tinyurl.com/pose-features) and [Object Features](https://tinyurl.com/object-features)

If you want to recreate our dataset curation pipeline you can do so in the following steps:

Step 1: Download [NTURGBD dataset](https://rose1.ntu.edu.sg/dataset/actionRecognition/),follow the steps to get the dataset.

Step 2: Download the action combination list we created [ACTION LIST]( https://tinyurl.com/data-curation).

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

Alternatively you can access our [TRAINING_DATA]( https://tinyurl.com/instruction-data) here if you want to skip the above process.

You can adapt the above process for your own ADL dataset curation with any ADL data just create your own action combinations like that of STEP 2.

---

## Quantitative Evaluation :🧪:

We introduce two new evaluation for ADL centric tasks -- ADLMCQ-AR & ADLMCQ-AF
---

## Qualitative Analysis :mag:

<p align="center">
  <img src="./llavidal/static/QA_example.png" alt="Qualitative Evaluation">
</p>  

---

## Acknowledgements :pray:

+ [LLaMA](https://github.com/facebookresearch/llama): A great attempt towards open and efficient LLMs!
+ Additional acknowledgements as needed.

If you're using LLAVIDAL in your research or applications, please cite using this BibTeX:
```bibtex
@inproceedings{Chakraborty2024LLAVIDAL,
    title={LLAVIDAL: Benchmarking Large LAnguage VIsion Models for Daily Activities of Living},
    author={Chakraborty, Rajatsubhra and Sinha, Arkaprava and Reilly, Dominick and Govind, Manish Kumar and Wang, Pu and Bremond, François and Das, Srijan},
    booktitle={},
    year={2024}
}
