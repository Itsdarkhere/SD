import os, stat
from typing import List

import torch
from pytorch_lightning import seed_everything
from diffusers import (
    StableDiffusionInpaintPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from cog import BasePredictor, Path, Input
# from PIL import Image


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-inpainting"

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
            cache_dir="pretrain/tokenizer",
            local_files_only=True,
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            cache_dir="pretrain/text_encoder",
            local_files_only=True,
        )

        # Load the learned concepts embeddings
        self.concepts = [
            'bedroom9000.pt',
            'bedroom9800.pt',
            'boardroom6000.pt',
            'boardroom10000.pt',
            'empty5000.pt',
            'living6000.pt',
            'living10000.pt',
            'office10000.pt',
            'office13200.pt',
            'privateoffice8200.pt',
            'privateoffice10000.pt',
        ]

        self.embeddings_dict = {}
        self.placeholder_tokens_dict = {}
        for concept in self.concepts:
            loaded_learned_embeds = torch.load(concept, map_location="cpu")
            embeds = next(iter(loaded_learned_embeds['string_to_param'].values()))
            self.embeddings_dict[concept] = embeds

            # Add tokens and update the text encoder for each concept
            concept_letter = concept[0]
            placeholder_token = ""
            for i, emb in enumerate(embeds):
                new_token = f"_{concept_letter}{i+1}"
                placeholder_token += new_token
                self.tokenizer.add_tokens(new_token)
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            
            for i, emb in enumerate(embeds):
                new_token = f"_{concept_letter}{i+1}"
                token_id = self.tokenizer.convert_tokens_to_ids(new_token)
                self.text_encoder.get_input_embeddings().weight.data[token_id] = emb

            self.placeholder_tokens_dict[concept] = placeholder_token

    def predict(
        self,
        concept: str = Input(
            choices=[
                'bedroom9000.pt',
                'bedroom9800.pt',
                'boardroom6000.pt',
                'boardroom10000.pt',
                'empty5000.pt',
                'living6000.pt',
                'living10000.pt',
                'office10000.pt',
                'office13200.pt',
                'privateoffice8200.pt',
                'privateoffice10000.pt',
            ],
            default="bedroom9000.pt",
            description="Choose a pretrained concept. The Placeholder is shown in <your-chosen-concept>.",
        ),
        prompt: str = Input(
            description="Input prompt with <your-chosen-concept>.",
            default="a <cat-toy> themed lunchbox",
        ),
        image: Path = Input(
            description="Inital image to generate variations of. Supproting images size with 512x512",
        ),
        mask: Path = Input(
            description="Black and white image to use as mask for inpainting over the image provided. White pixels are inpainted and black pixels are preserved",
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            # choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            # choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=8,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps for each image generated from the prompt",
            ge=1,
            le=500,
            default=50,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        image = Image.open(image).convert("RGB").resize((width, height))
        extra_kwargs = {
            "mask_image": Image.open(mask).convert("RGB").resize(image.size),
            "image": image
        }

        # Separate the token and the embed
        # Use the stored embeddings for the chosen concept
        embeddings = self.embeddings_dict[concept]
        placeholder_token = self.placeholder_tokens_dict[concept]

        # Insert the placeholder token into the prompt
        prompt = prompt.replace("<your-chosen-concept>", placeholder_token)

        # placeholder_token Needs to be in the prompt
        print("loading StableDiffusionInpaintPipeline with updated tokenizer and text_encoder")
        
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            cache_dir="pretrain/diffusers-cache",
            local_files_only=True,
            # torch_dtype=torch.float16,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        ).to("cuda")

        pipeline.scheduler = make_scheduler(scheduler, pipeline.scheduler.config)

        print(f"{placeholder_token} placeholder_token")
        generator = torch.Generator("cuda").manual_seed(seed)

        images = pipeline(
            prompt=[prompt] * num_outputs,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **extra_kwargs,
        ).images

        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]