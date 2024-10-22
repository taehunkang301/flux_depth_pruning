import os
import re
import json
import time
from dataclasses import dataclass
from glob import iglob

import numpy as np

import torch
import torch.nn.functional as F

from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
from transformers import pipeline

NSFW_THRESHOLD = 0.85

@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting seed to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options

def compute_per_layer_score(tensor_list: list[torch.Tensor], mode="sim") -> np.ndarray:

    if mode == "sim":
        return compute_cosine_similarities(tensor_list=tensor_list)
    else:
        print("Not Implemented Option")
        raise


def compute_cosine_similarities(tensor_list: list[torch.Tensor]) -> np.ndarray:
    num_blocks = len(tensor_list)
    similarities = np.zeros(num_blocks - 1)

    for i in range(num_blocks - 1):
        input_i = tensor_list[i].flatten(1)
        output_i = tensor_list[i+1].flatten(1)  # Shape: [batch_size, features]
        sim = F.cosine_similarity(input_i, output_i, dim=1).mean().item()
        similarities[i] = sim

    return similarities


def load_prompts(file_path, num_prompts=100):
    prompts = []
    with open(file_path, 'r') as f:
        for _ in range(num_prompts):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split('\t')
            if len(parts) == 2:
                prompts.append(parts[1])
    return prompts


def main(
    name: str = "flux-dev",
    width: int = 256,
    height: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 50,
    guidance: float = 3.5,
    offload: bool = False,
    output_dir: str = "output",
    data_file: str = "/workspace/repo/flux/data/cc12m_1000.txt",
    num_prompts: int = 100,
    score_mode: str = "sim",
    skip_cnt: int = 8
):
    """
    Main function to run the FLUX model and compute cosine similarities between block outputs.
    """
    torch_device = torch.device(device)
    dtype = torch.bfloat16  # Adjust as needed

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the first 100 prompts from CC12M dataset
    prompts = load_prompts(data_file, num_prompts=num_prompts)
    
    # Load models
    print("Loading models...")
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)
    
    # Initialize random generator
    rng = torch.Generator(device="cpu")
    
    # Adjust height and width to be multiples of 16
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    skip_idx = []
    for _ in range(skip_cnt + 1):

        # Initialize a list to store all similarities
        prompts_score = []
        
        # Traverse target prompts
        for prompt_idx, prompt in enumerate(prompts):

            #############


            # On-going
            # TODO
            # Gather all-prompts at once and then compute max idx
            # How about timestep difference?


            #############

            # Prepare options
            opts = SamplingOptions(
                prompt=prompt,
                width=width,
                height=height,
                num_steps=num_steps,
                guidance=guidance,
                seed=0,  # You can set a specific seed if needed
            )
            
            # Generate initial noise
            x = get_noise(
                1,  # batch size of 1
                opts.height,
                opts.width,
                device=torch_device,
                dtype=dtype,
                seed=opts.seed,
            )
            
            if offload:
                ae = ae.cpu()
                torch.cuda.empty_cache()
                t5, clip = t5.to(torch_device), clip.to(torch_device)
            
            # Prepare inputs
            inp = prepare(t5, clip, x, prompt=opts.prompt)
            timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))
            
            # Offload models if necessary
            if offload:
                t5, clip = t5.cpu(), clip.cpu()
                torch.cuda.empty_cache()
                model = model.to(torch_device)
            
            # Denoise initial noise and collect outputs
            x, outputs_per_timestep = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance, skip_idx=skip_idx)
            
            # Compute cosine similarities per timestep
            for timestep_idx in range(len(timesteps) - 1):
                timestep_data = {
                    'prompt': prompt,
                    'timestep': timestep_idx,
                    'block_score': None,
                }
                
                # DoubleStreamBlocks image outputs
                block_score = compute_per_layer_score(outputs_per_timestep[timestep_idx], mode=score_mode)
                timestep_data['redundant_idx'] = max_idx
                timestep_data['block_score'] = block_score.tolist()
                
                # Append the data
                prompts_score.append(timestep_data)
            
            # Offload model, load autoencoder to GPU if necessary
            if offload:
                model.cpu()
                torch.cuda.empty_cache()
                ae.decoder.to(x.device)

        block_scores = [d['block_score'] for d in prompts_score if d['block_score'] is not None]
        average_block_score = np.mean(block_scores, axis=0)
        target_idx = np.argmax(average_block_score)
        skip_idx.append(target_idx)
        
    print("Skip idx:", skip_idx)

    # Save all similarities to a JSON file
    similarities_file = os.path.join(output_dir, f'block_score_{score_mode}.json')
    print(f"\nSaving redundancy score ({score_mode}) to {similarities_file}")
    with open(similarities_file, 'w') as f:
        json.dump(all_similarities, f)
    
    print("Processing complete.")

def app():
    Fire(main)


if __name__ == "__main__":
    app()
