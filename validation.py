import metric
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from torch import autocast
import argparse
import json
import os

def process_color(color):
	red, green, blue = bytes.fromhex(color)
	return (red, green, blue)

def compute_quality(model_name):
    controlnet = ControlNetModel.from_pretrained(f'MexFoundation/{model_name}', torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'MexFoundation/map_diffusion-1.2', controlnet=controlnet, torch_dtype=torch.float32, safety_checker = None, requires_safety_checker=False
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    colors = json.loads(open('colors.json').read())
    colors = { process_color(key): value for key, value in colors.items() }
    colors2 = json.loads(open('colors2.json').read())
    colors2 = { process_color(key): value for key, value in colors2.items() }

    tot_metric = 0
    cnt = 0

    for filename in os.listdir('tokyo5'):
        print(f'Iteration {cnt}, file {filename}')
        image0 = load_image(f'tokyo5/{filename}')
        image = image0.crop([0, 0, 512, 512])
        image1 = image0.crop([600, 0, 600 + 512, 512])
        num_samples = 1
        generator = [torch.Generator(device='cpu').manual_seed(2) for i in range(num_samples)]

        with autocast('cuda'):
            images = pipe(['mmap'] * num_samples, image, negative_prompt=[""] * num_samples, generator=generator, num_inference_steps=20).images

        image2 = images[0]
        cnt += 1
        metric_val = metric.metric(image1.load(), image2.load(), colors, colors2)[0]
        print('Metric:', f'{metric_val:.2%}')
        tot_metric += metric_val

    avg_metric = tot_metric / cnt
    print('Average metric: ', f'{avg_metric:.2%}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model')
    args = parser.parse_args()
    compute_quality(args.model)
