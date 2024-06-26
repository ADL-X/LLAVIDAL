import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import torch
import transformers
from torch.utils.data import Dataset
from llavidal.train.llava_trainer import LlavidalTrainer
from llavidal import video_conversation as conversation_lib
from llavidal.model import *
import torch.distributed as dist
from llavidal.constants import *
import pickle
import numpy as np
import os

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_vid_start_end: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    sep_video_conv_front: bool = False
    video_token_len: int = 0
    video_folder: Optional[str] = field(default=None)
    object_folder: Optional[str] = field(default=None)

    #pose_folder: Optional[str] = field(default=None)  # Add this line
    frame_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    dist.barrier()
    state_dict = trainer.model.state_dict()
    trainer._save(output_dir, state_dict=state_dict)  # noqa

# def load_object_features(object_file):
#     """Load object features from a pickle file and determine the token length per object."""
#     with open(object_file, 'rb') as file:
#         object_features = pickle.load(file)
    
#     # Assuming the shape of object_features is [num_objects, features_per_object]
#     num_objects = object_features.shape[0]
#     object_token_len = 8  # Define how many tokens represent one object, if static

#     return object_features, num_objects, object_token_len

def generate_fake_pose_features(min_len=75, max_len=200, dim=216):
        pose_len = np.random.randint(min_len, max_len + 1)
        return np.random.randn(pose_len, dim)

# def preprocess_object_features(sources, object_token_len, num_objects, tokenizer):
#     tokenized_objects = []
#     for source in sources:
#         object_features = []
#         for obj_idx in range(num_objects):
#             start_idx = obj_idx * object_token_len * 512
#             end_idx = start_idx + (object_token_len * 512)
#             print(start_idx,end_idx)
#             #object_feature = source['object'][start_idx:end_idx]
#             object_token = tokenizer.encode(DEFAULT_OBJECT_START_TOKEN + DEFAULT_OBJECT_PATCH_TOKEN * object_token_len + DEFAULT_OBJECT_END_TOKEN, add_special_tokens=False)
#             object_features.extend(object_token)
#         tokenized_objects.append(object_features)
#     return tokenized_objects

    

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        multimodal_cfg: dict,
        cur_video_token_len: int,
        cur_object_token_len: int
) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']
    
    video_token_len = cur_video_token_len
    object_token_len = cur_object_token_len
    num_objects = object_token_len//8 
    if not is_multimodal:
        return sources

    for source in sources:

        for source in sources:
         if multimodal_cfg['sep_video_conv_front']:
            assert DEFAULT_VIDEO_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_VIDEO_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_VIDEO_TOKEN + conversation_lib.default_conversation.sep + \
                                 conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        for sentence in source:
            replace_video_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
            if multimodal_cfg['use_vid_start_end']:
                replace_video_token = DEFAULT_VID_START_TOKEN + replace_video_token + DEFAULT_VID_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, replace_video_token)



        for sentence in source:
            if num_objects > 1:
                sentence["value"] = sentence["value"].replace(DEFAULT_OBJECT_TOKEN, DEFAULT_OBJECT_TOKEN*num_objects)

            replace_token = DEFAULT_OBJECT_PATCH_TOKEN * (object_token_len // num_objects)
            if multimodal_cfg['use_vid_start_end']: # use video config for object as well
                replace_token = DEFAULT_OBJECT_START_TOKEN + replace_token + DEFAULT_OBJECT_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_OBJECT_TOKEN, replace_token)    

            # if 'object' in sentence:
            #     num_objects = len(sentence['object']) // (object_token_len * 512)
            #     #object_features = preprocess_object_features(sentence['object'], object_token_len, num_objects, tokenizer)
            #     replace_object_token = "".join([
            #         DEFAULT_OBJECT_START_TOKEN + DEFAULT_OBJECT_PATCH_TOKEN * object_token_len + DEFAULT_OBJECT_END_TOKEN
            #         for _ in range(num_objects)
            #     ])
            #     sentence["value"] = sentence["value"].replace(DEFAULT_OBJECT_TOKEN, replace_object_token)

         

        # if multimodal_cfg['sep_video_conv_front']: # Dominick: In default cfg this is false
        #     #raise NotImplementedError("Not implemented for poses yet")
        #     assert DEFAULT_VIDEO_TOKEN in source[0]['value']
        #     source[0]['value'] = source[0]['value'].replace(DEFAULT_VIDEO_TOKEN, '').strip()
        #     source[0]['value'] = DEFAULT_VIDEO_TOKEN + conversation_lib.default_conversation.sep + \
        #                          conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        # for sentence in source:
        #     replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
        #     if multimodal_cfg['use_vid_start_end']:
        #         replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN
        #     sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, replace_token)

        # for sentence in source:
        #     replace_token = DEFAULT_POSE_PATCH_TOKEN * pose_token_len
        #     if multimodal_cfg['use_vid_start_end']: # use video config for pose as well
        #         replace_token = DEFAULT_POSE_START_TOKEN + replace_token + DEFAULT_POSE_END_TOKEN
        #     sentence["value"] = sentence["value"].replace(DEFAULT_POSE_TOKEN, replace_token)

    return sources


def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        # breakpoint()
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # breakpoint()
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # breakpoint()
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        # breakpoint()
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    # breakpoint() 
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)

    assert False, "Unexpected" # conversation_lib.default_conversation.version is v1 when I checked (Dominick)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        #breakpoint()
        if 'video' in sources[0]: # Assuming pose is only present if a video is present
            video_file = self.list_data_dict[i]['video']
            #breakpoint()
            video_folder = self.multimodal_cfg['video_folder']
            with open(f"{video_folder}/{video_file}", "rb") as f:
                features = pickle.load(f)
            print(f"READING VIDEO FROM {video_folder}/{video_file} #########################################################################")
        if 'object' in sources[0]: # Assuming object data is present
            #breakpoint()
            object_file = self.list_data_dict[i]['object']
            object_folder = self.multimodal_cfg['object_folder']
            if os.path.exists(f"{object_folder}/{object_file}"):
                print(f"READING OBJECT FROM {object_file} #############################")
                with open(f"{object_folder}/{object_file}", "rb") as f:
                    object_features = pickle.load(f)
                    object_features = object_features.reshape(-1, 512)  # Reshape to [8, 512] if it's not already
                    print("Loaded object features shape:", object_features.shape)
            else:
                # If the object file does not exist, create a zero tensor as a placeholder
                print(f"No object data found for {object_file}, using zero tensor as placeholder.")
                object_features = torch.zeros((8, 512), dtype=torch.float32)  # Create a zero tensor of shape [8, 512
        
            cur_video_token_len = 356  
            cur_object_token_len = object_features.shape[0] 

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.multimodal_cfg, cur_video_token_len, cur_object_token_len)
        #breakpoint()
        data_dict = preprocess(
            sources,
            self.tokenizer)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # breakpoint()    

        # video exist in the data
        if 'video' in self.list_data_dict[i]:
            data_dict["video"] = features
            #data_dict["pose"] = torch.from_numpy(pose_pad).to(torch.bfloat16)
        
        if os.path.exists(f"{object_folder}/{object_file}"):
            data_dict["object"] = torch.tensor(object_features)
        

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'video' in instances[0]: # TODO: Assuming pose is only present if video is present
            features = [torch.tensor(instance['video']) for instance in instances]
            #pose_features = [torch.tensor(instance['pose']) for instance in instances]
            #breakpoint()
            if all(x is not None and x.shape == features[0].shape for x in features):
                batch['video_spatio_temporal_features'] = torch.stack(features)
            else:
                batch['video_spatio_temporal_features'] = features


            # Handle object features if present
        if any('object' in instance for instance in instances):
            # Filter out instances that have object features
            object_features = [torch.tensor(instance['object']) for instance in instances if 'object' in instance]
            if object_features:
                # Pad sequences to handle variable lengths
                batch['object_features'] = torch.nn.utils.rnn.pad_sequence(object_features, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            else:
                # If no valid object features are present, you might add a placeholder or handle it differently
                logging.info("No valid object features found among the batch instances.")
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                multimodal_cfg=dict(
                                    is_multimodal=data_args.is_multimodal,
                                    sep_video_conv_front=data_args.sep_video_conv_front,
                                    video_token_len=data_args.video_token_len,
                                    video_folder=data_args.video_folder,
                                    object_folder=data_args.object_folder,  # Pass the pose folder here
                                    frame_aspect_ratio=data_args.frame_aspect_ratio,
                                    use_vid_start_end=getattr(data_args, 'mm_use_vid_start_end', False)))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = LLAVIDALLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        # torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float,
    )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    model_vision_dict = model.get_model().initialize_vision_modules(
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
    )
    vision_config = model_vision_dict['vision_config']

    data_args.video_token_len = model_vision_dict['video_token_len']
    data_args.is_multimodal = True

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model.config.mm_use_vid_start_end = data_args.mm_use_vid_start_end = model_args.mm_use_vid_start_end
    vision_config.use_vid_start_end = training_args.use_vid_start_end = model_args.mm_use_vid_start_end
    model.config.sep_video_conv_front = data_args.sep_video_conv_front
    model.initialize_vision_tokenizer(mm_use_vid_start_end=model_args.mm_use_vid_start_end, tokenizer=tokenizer,
                                      device=training_args.device, tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                                      pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(
                    len(params_no_grad), params_no_grad))
            else:
                print(
                    '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
                        len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)

                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args) # this is the dataset
    training_args.report_to = []
    # training_args.max_steps = 10
    trainer =LlavidalTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()