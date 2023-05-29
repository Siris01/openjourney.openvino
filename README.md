# openjournye.openvino

Implementation of Text-To-Image generation using openjourney on Intel CPU or GPU.


## Install requirements

* Set up and update PIP to the highest version
* Install OpenVINO™ Development Tools 2022.3.0 release with PyPI
* Download requirements

```bash
python -m pip install --upgrade pip
pip install openvino-dev[onnx,pytorch]==2022.3.0
pip install -r requirements.txt
```

## Generate image from text description

```bash
usage: run.py [-h] [--model MODEL] [--device DEVICE] [--seed SEED] [--beta-start BETA_START] [--beta-end BETA_END] [--beta-schedule BETA_SCHEDULE]
               [--num-inference-steps NUM_INFERENCE_STEPS] [--guidance-scale GUIDANCE_SCALE] [--eta ETA] [--tokenizer TOKENIZER] [--prompt PROMPT] [--params-from PARAMS_FROM]
               [--init-image INIT_IMAGE] [--strength STRENGTH] [--mask MASK] [--output OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model name
  --device DEVICE       inference device [CPU, GPU]
  --seed SEED           random seed for generating consistent images per prompt
  --beta-start BETA_START
                        LMSDiscreteScheduler::beta_start
  --beta-end BETA_END   LMSDiscreteScheduler::beta_end
  --beta-schedule BETA_SCHEDULE
                        LMSDiscreteScheduler::beta_schedule
  --num-inference-steps NUM_INFERENCE_STEPS
                        num inference steps
  --guidance-scale GUIDANCE_SCALE
                        guidance scale
  --eta ETA             eta
  --tokenizer TOKENIZER
                        tokenizer
  --prompt PROMPT       prompt
  --params-from PARAMS_FROM
                        Extract parameters from a previously generated image.
  --init-image INIT_IMAGE
                        path to initial image
  --strength STRENGTH   how strong the initial image should be noised [0.0, 1.0]
  --mask MASK           mask of the region to inpaint on the initial image
  --output OUTPUT       output image name
```

## Acknowledgements

* Original code for StableDiffusion using openvino: https://github.com/bes-dev/stable_diffusion.openvino
* Original implementation of OpenJourney: https://huggingface.co/prompthero/openjourney
* diffusers library: https://github.com/huggingface/diffusers

## Disclaimer

The authors are not responsible for the content generated using this project.
Please, don't use this project to produce illegal, harmful, offensive etc. content.
