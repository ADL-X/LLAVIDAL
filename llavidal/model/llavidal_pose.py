from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"

DEFAULT_POSE_TOKEN = "<human_pose>"
DEFAULT_POSE_PATCH_TOKEN = "<pose_patch>"
DEFAULT_POSE_START_TOKEN = "<pose_start>"
DEFAULT_POSE_END_TOKEN = "<pose_end>"


class VisionConfig:
    def __init__(self):
        self.frame_size = 224
        self.patch_size = 14
        self.hidden_size = 1024 # the shape of the features from the vision encoder
        self.hidden_size_pose = 216 # the shape of the features from the vision encoder
        self.use_vid_start_end = None
        self.vid_start_token = None
        self.vid_end_token = None
        self.vid_patch_token = None


class LLAVIDALConfig(LlamaConfig):
    model_type = "LLAVIDAL"


class LLAVIDALLlamaModel(LlamaModel):
    config_class = LLAVIDALConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):  # TODO: Remove unused params
        super(LLAVIDALLlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_config = VisionConfig()

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            self.mm_projector_forpose = nn.Linear(self.vision_config.hidden_size_pose, config.hidden_size) # ! There are a lot of lines doing things to mm_projector (above). I am not implementing them for mm_projector_forpose for now

    def initialize_vision_modules(self, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        vision_config = self.vision_config
        num_patches = (vision_config.frame_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_hidden_size_pose = vision_config.hidden_size_pose

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})
            self.mm_projector_forpose.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            video_token_len=num_patches,
            vision_config=vision_config
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            pose_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (input_ids.shape[1] != 1 or self.training) and video_spatio_temporal_features is not None:

            video_features = self.mm_projector(video_spatio_temporal_features)
            dummy_video_features = torch.zeros(video_features.shape[1], 1024, device=inputs_embeds.device,
                                               dtype=inputs_embeds.dtype)
            dummy_video_features = self.mm_projector(dummy_video_features)

            pose_features_projected = self.mm_projector_forpose(pose_features)

            new_input_embeds = []
            cur_video_idx = 0

            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.vision_config.vid_patch_token).sum() == 0:
                    raise NotImplementedError("Didnt expect this, the video was empty. If this error is raised then implement this for pose aswell")
                    # Multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_video_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_video_idx += 1
                    continue
                
                '''
                Initially, the input_embeds are the embeddings of the tokens in the input_ids. This means that the special video and pose tokens (e.g., <vid_patch> and <pose_start>) were processed like words (and the embeddings are the representations from the model).

                Below, we replace the embeddings of the video and pose tokens with the actual video and pose features. We do this by finding the positions of the video and pose tokens in the input_ids, and then replacing the embeddings of the tokens at those positions with the video and pose features.
                '''
                if self.vision_config.use_vid_start_end: # This is true in the config we use
                    if (cur_input_ids == self.vision_config.vid_start_token).sum() != (
                            cur_input_ids == self.vision_config.vid_end_token).sum():
                        raise ValueError("The number of video start tokens and video end tokens should be the same.")
                    video_start_tokens = torch.where(cur_input_ids == self.vision_config.vid_start_token)[0]
                    for video_start_token_pos in video_start_tokens:
                        cur_video_features = video_features[cur_video_idx].to(device=cur_input_embeds.device)
                        cur_pose_features = pose_features_projected[cur_video_idx].to(device=cur_input_embeds.device)

                        num_patches = cur_video_features.shape[0] # the number of video features
                        num_pose_patches = cur_pose_features.shape[0] # the number of pose features

                        if cur_input_ids[video_start_token_pos + num_patches + 1] != self.vision_config.vid_end_token:
                            raise ValueError("The video end token should follow the video start token.")
                        if orig_embeds_params is not None:
                            # ! Get pose embeddings relative to video embedding
                            pose_start_token_idx = video_start_token_pos + num_patches + 2
                            pose_end_token_idx = pose_start_token_idx + num_pose_patches + 1
                            #breakpoint()
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos].detach(), # everything before vid 
                                                                cur_input_embeds[
                                                                video_start_token_pos:video_start_token_pos + 1], # vid_start token
                                                                cur_video_features, # the video features
                                                                cur_input_embeds[
                                                                    video_start_token_pos + num_patches
                                                                    + 1:video_start_token_pos
                                                                    + num_patches + 2], # vid_end token
                                                                cur_input_embeds[pose_start_token_idx:pose_start_token_idx + 1], # pose_start token
                                                                cur_pose_features, # the pose features
                                                                cur_input_embeds[pose_end_token_idx:pose_end_token_idx + 1], # pose_end token
                                                                cur_input_embeds[
                                                                pose_end_token_idx + 1:].detach()), # everything after pose
                                                                dim=0)
                            # ! The shapes arent the same, im not sure if they should be. TODO: Check later
                            # cur_new_input_embeds_TEST = torch.cat((cur_input_embeds[:video_start_token_pos + 1], # everything before vid
                            #                                     cur_video_features, # the video features
                            #                                     cur_input_embeds[video_start_token_pos + num_patches + 1:pose_start_token_idx], # everything after vid and before pose
                            #                                     cur_pose_features, # the pose features
                            #                                     cur_input_embeds[pose_end_token_idx + 1:]), dim=0) # everything after pose
                            # breakpoint()
                        else:
                            pose_start_token_idx = video_start_token_pos + num_patches + 2
                            pose_end_token_idx = pose_start_token_idx + num_pose_patches + 1
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos + 1], # everything before vid
                                                                cur_video_features, # the video features
                                                                cur_input_embeds[video_start_token_pos + num_patches + 1:pose_start_token_idx], # everything after vid and before pose
                                                                cur_pose_features, # the pose features
                                                                cur_input_embeds[pose_end_token_idx + 1:]), dim=0) # everything after pose
                        cur_video_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_video_features = video_features[cur_video_idx]
                    num_patches = cur_video_features.shape[0]
                    if (cur_input_ids == self.vision_config.vid_patch_token).sum() != num_patches:
                        raise ValueError(
                            "The number of video patch tokens should be the same as the number of video patches.")
                    masked_indices = torch.where(cur_input_ids == self.vision_config.vid_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patches,
                                                       device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The video patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(),
                                                          cur_video_features,
                                                          cur_input_embeds[mask_index_start + num_patches:].detach()),
                                                         dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_video_features,
                                                          cur_input_embeds[mask_index_start + num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_video_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(LLAVIDALLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class LLAVIDALLlamaForCausalLM(LlamaForCausalLM):
    config_class = LLAVIDALConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LLAVIDALLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            pose_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            video_spatio_temporal_features=video_spatio_temporal_features,
            pose_features=pose_features
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
                "pose_features": kwargs.get("pose_features", None)
            }
        )
        breakpoint()
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_vid_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.get_model().vision_config
        vision_config.use_vid_start_end = mm_use_vid_start_end
        tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_POSE_PATCH_TOKEN], special_tokens=True) # ! Add pose tokens
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_vid_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_POSE_START_TOKEN, DEFAULT_POSE_END_TOKEN], special_tokens=True) # ! Add pose tokens
            self.resize_token_embeddings(len(tokenizer))
            vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])
            vision_config.pose_start_token, vision_config.pose_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_POSE_START_TOKEN, DEFAULT_POSE_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
        vision_config.pose_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_POSE_PATCH_TOKEN])[0]


AutoConfig.register("LLAVIDAL", LLAVIDALConfig)
AutoModelForCausalLM.register(LLAVIDALConfig, LLAVIDALLlamaForCausalLM)