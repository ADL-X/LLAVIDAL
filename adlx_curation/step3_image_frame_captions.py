from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import argparse, json, torch

import random, tqdm, time, glob, cv2, PIL, os
import numpy as np
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--proc_num', type=int, default=0, help='The process number of this job')
parser.add_argument('--num_procs', type=int, default=1, help='The total number of processes to run')
parser.add_argument('--stitched_vid_path', type=str, required=True)
parser.add_argument('--save_every', type=int, default=100)
args = parser.parse_args()

DEVICE = f'cuda:{args.proc_num}'

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')

model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-chat-hf',
    low_cpu_mem_usage=True,
    # load_in_4bit=True,
    trust_remote_code=True,
    # device_map='auto',
    torch_dtype=torch.float16,
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16
    # )
).eval().to(DEVICE)

query = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Give a detailed description of the actions happening and describe the image, include motions and the objects interacted by the person. Do not provide any coordinates from the image ASSISTANT:"

gen_kwargs = {
    # "max_length": 2048,
    "max_new_tokens": 256,
    "do_sample": False} # "temperature": 0.9


# Load video paths
vids = glob.glob(f'{args.stitched_vid_path}/*.mp4')
random.shuffle(vids) # seed is set above, so multiproc is okay

partition_size = len(vids) // args.num_procs
if args.proc_num == args.num_procs - 1: # last process
    vids = vids[args.proc_num * partition_size:]
else:
    vids = vids[args.proc_num * partition_size: (args.proc_num + 1) * partition_size]

# Create json file
output_json_path = f'./cogvlm_{args.proc_num}.json'
if not os.path.exists(output_json_path):
    with open(output_json_path, "w") as f:
        json.dump({}, f)  # Create an empty JSON structure

with open(output_json_path, "r") as f:
    progress_data = json.load(f)

batch = {}
save_every = args.save_every

caption_rate_fps = 0.5
DEVICE = model.device

iterator = tqdm.tqdm(enumerate(vids), total=len(vids))
for i, vid in iterator:
    vid_name = vid.split('/')[-1][:-4]

    # check video is not corrupt. greater than 150 bytes
    if os.path.getsize(vid) < 150:
        continue

    if vid_name in progress_data:
        continue

    cap = cv2.VideoCapture(vid)
    num_frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    num_captions = int(num_frames / fps / caption_rate_fps)
    num_captions = 1 if num_captions == 0 else num_captions

    frame_idxs = np.linspace(0, num_frames - 1, num_captions, dtype=int)

    batch[vid_name] = []

    try:
        for frame_idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            image = PIL.Image.fromarray(frame[..., ::-1]).convert('RGB')

            input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])

            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                'images': [[input_by_model['images'][0].to(DEVICE).to(torch.float16)]] if image is not None else None,
            }

            with torch.no_grad():
                s = time.time()
                outputs = model.generate(**inputs, **gen_kwargs)
                # print('Forward time:', time.time() - s)

                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(outputs[0])
                response = response.split("</s>")[0]
                batch[vid_name].append(response)
            
            iterator.set_description(f'Caption {len(batch[vid_name])}/{num_captions}')
    except Exception as e:
        print(f'Error processing {vid_name}: {e}')
        batch[vid_name] = f'failed to get caption with exception {e}'

    if len(batch) >= save_every or i == len(vids) - 1:
        progress_data.update(batch)

        with open(output_json_path, "w") as f:
            json.dump(progress_data, f, indent=4)

        batch = {}

    cap.release()