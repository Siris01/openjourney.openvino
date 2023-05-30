# -- coding: utf-8 --`
import argparse
import random
# engine
from engine import StableDiffusionEngine
# scheduler
from diffusers import PNDMScheduler
# utils
import cv2
import numpy as np
from openvino.runtime import Core


def main(args):
    if args.seed is None:
        args.seed = random.randint(0, 2**30)
    np.random.seed(args.seed)

    scheduler = PNDMScheduler(
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        skip_prk_steps=True,
        num_train_timesteps=1000,
        set_alpha_to_one=False,
        steps_offset=1,
        trained_betas=None,
    )
    engine = StableDiffusionEngine(
        model=args.model,
        scheduler=scheduler,
        tokenizer=args.tokenizer,
        device=args.device,
        height=args.h,
        width=args.w,
    )
    image = engine(
        prompt=f"{args.prompt} mdjrny-v4 style",
        init_image=None if args.init_image is None else cv2.imread(
            args.init_image),
        mask=None if args.mask is None else cv2.imread(args.mask, 0),
        strength=args.strength,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
    )
    cv2.imwrite(args.output, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument(
        "--model", type=str, default="Siris/openjourney-openvino-model", help="model name")
    # inference device
    parser.add_argument("--device", type=str, default="CPU",
                        help=f"inference device [{', '.join(Core().available_devices)}]")
    # randomizer params
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for generating consistent images per prompt")
    # scheduler params
    parser.add_argument("--beta-start", type=float,
                        default=0.00085, help="PNDMScheduler::beta_start")
    parser.add_argument("--beta-end", type=float, default=0.012,
                        help="PNDMScheduler::beta_end")
    parser.add_argument("--beta-schedule", type=str, default="scaled_linear",
                        help="PNDMScheduler::beta_schedule")
    # diffusion params
    parser.add_argument("--num-inference-steps", type=int,
                        default=32, help="num inference steps")
    parser.add_argument("--guidance-scale", type=float,
                        default=7.5, help="guidance scale")
    parser.add_argument("--eta", type=float, default=0.0, help="eta")
    # tokenizer
    parser.add_argument("--tokenizer", type=str,
                        default="openai/clip-vit-large-patch14", help="tokenizer")
    # prompt
    parser.add_argument("--prompt", type=str, help="prompt")
    # Parameter re-use:
    parser.add_argument("--params-from", type=str, required=False,
                        help="Extract parameters from a previously generated image.")
    # img2img params
    parser.add_argument("--init-image", type=str,
                        default=None, help="path to initial image")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="how strong the initial image should be noised [0.0, 1.0]")
    # inpainting
    parser.add_argument("--mask", type=str, default=None,
                        help="mask of the region to inpaint on the initial image")
    # output name
    parser.add_argument("--output", type=str,
                        default="output.png", help="output image name")

    parser.add_argument("--h", type=int, default=512, help="height")
    parser.add_argument("--w", type=int, default=512, help="width")

    args = parser.parse_args()
    main(args)
