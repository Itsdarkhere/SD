import os, stat
from typing import List

import torch
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from huggingface_hub import hf_hub_download
from cog import BasePredictor, Path, Input


with open("concepts.txt") as infile:
    CONCEPTS = [line.rstrip() for line in infile]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

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

    def predict(
        self,
        concept: str = Input(
            choices=CONCEPTS,
            default="sd-concepts-library/cat-toy: <cat-toy>",
            description="Choose a pretrained concept. The Placeholder is shown in <your-chosen-concept>.",
        ),
        prompt: str = Input(
            description="Input prompt with <your-chosen-concept>.",
            default="a <cat-toy> themed lunchbox",
        ),
        num_outputs: int = Input(
            description="Number of images to output", choices=[1, 4], default=1
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
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        seed_everything(seed)

        embeds_path = './bonzi.pt'
        placeholder = '<bonzi>'
        
        # Load the learned concept
        loaded_learned_embeds = torch.load(embeds_path, map_location="cpu")

        # Separate the token and the embed
        # trained_token = list(loaded_learned_embeds.keys())[0]
        # embeds = loaded_learned_embeds[trained_token]
        string_to_token = loaded_learned_embeds['string_to_token']
        string_to_param = loaded_learned_embeds['string_to_param']
        trained_token = list(string_to_token.keys())[0]
        embeds = string_to_param[trained_token]
        embeds = embeds.detach()
        embeds = embeds[1]

        # Convert the embed to the same dtype as the text_encoder
        dtype = self.text_encoder.get_input_embeddings().weight.dtype
        embeds.to(dtype)

        # Add the token to the tokenizer
        token = placeholder
        num_added_tokens = self.tokenizer.add_tokens(token)
        print(f"{num_added_tokens} new tokens added.")

        # Make sure a token was added
        # if num_added_tokens == 0:
        #     raise ValueError(f"The tokenizer already contains the token {token}.")

        # Resize token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Get id for the token and assign the embedding to it
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

        print("loading StableDiffusionPipeline with updated tokenizer and text_encoder")
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            cache_dir="pretrain/diffusers-cache",
            local_files_only=True,
            # torch_dtype=torch.float16,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        ).to("cuda")

        print("Generating images with the learned concept")
        images = pipeline(
            prompt=[prompt] * num_outputs,
            width=512,
            height=512,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images

        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
