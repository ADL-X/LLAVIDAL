# LLAVIDAL
#  LLAVIDAL 🏃👩‍🦯‍➡️🗨️

### LLAVIDAL: Towards Detailed Video Understanding via Large Vision and Language Models

#### [Rajatsubhra Chakraborty](https://chakrabortyrajatsubhra.github.io)<sup>1</sup>* , [Arkaprava Sinha](https://www.linkedin.com/in/arkaprava-sinha)<sup>1</sup>* , [Dominick Reilly](https://dominick-reilly.github.io/)<sup>1</sup>* , [Manish Kumar Govind](https://sites.google.com/view/manishkumargovind/home)<sup>1</sup>, [Pu Wang](https://webpages.charlotte.edu/pwang13/)<sup>1</sup>,[Francois Bremond](http://www-sop.inria.fr/members/Francois.Bremond/)<sup>2</sup> and [Srijan Das](https://srijandas07.github.io)<sup>1</sup>
\* Equally contributing first authors

##### Affiliations:
<sup>1</sup> University of North Carolina at Charlotte  
<sup>2</sup> INRIA, Université Côte d’Azur


## :loudspeaker: Latest Updates
Placeholder for updates

---

## Online Demo :computer:

Placeholder for online demo description

---

## LLAVIDAL Overview 👁️:

LLAVIDAL (Large LAnguage VIsion model for Daily Activities of Living) is a multimodal model designed to understand and generate meaningful conversations about activities of daily living (ADL) performed by humans in videos. Its architecture integrates multiple modalities, including video, 3D human poses, and object interaction cues, with a large language model (LLM). Here's an overview of LLAVIDAL's architecture:

Video Encoder: Input videos are encoded using a pre-trained vision-language model (VLM) such as CLIP-L/14 to obtain frame-level embeddings. These embeddings are then aggregated along temporal and spatial dimensions to generate video-level features.
Pose Encoder (PoseLM): 3D human pose sequences are processed using a pose-language model called PoseCLIP, which consists of a pose backbone and a CLIP text encoder. The pose features are aligned with the language domain through training.
Object Cues (ObjectLM): Relevant objects in the video are detected using a pre-trained object detection model (BLIP2) and localized using an open-vocabulary object localization model (OWLv2). Object features are extracted from the localized image regions.
Projection Layers: The video, pose, and object features are projected into the LLM's embedding space using separate linear projection layers (T_v, T_p, T_o) to align them with the LLM's input space.
LLM Integration: The projected video, pose, and object features are concatenated with tokenized text queries and fed into a frozen LLM (Vicuna language decoder). The LLM is fine-tuned on instructional language-vision data (ADL-X dataset) using an autoregressive training objective.

During training, LLAVIDAL learns to align the video, pose, and object features with the LLM's embedding space. The model is trained on the ADL-X dataset, which contains video-text pairs, 3D poses, and action-conditioned object trajectories.
At inference time, LLAVIDAL takes a video as input and generates meaningful conversations about the ADL performed in the video. It leverages the aligned video, pose, and object features to provide detailed and contextually relevant responses.
The integration of pose and object cues in LLAVIDAL enables it to better understand the fine-grained actions, human-object interactions, and temporal relationships present in ADL videos compared to models that rely solely on video features.

---

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

## Installation :wrench:

Placeholder for installation instructions

---

## Running Demo Offline :cd:

Placeholder for running demo offline instructions

---

## Training :train:

Placeholder for training instructions

---

## Video Instruction Dataset :open_file_folder:

Placeholder for video instruction dataset description

---

## Quantitative Evaluation :bar_chart:

Placeholder for quantitative evaluation description

---

## Qualitative Analysis :mag:

Placeholder for qualitative analysis description

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
