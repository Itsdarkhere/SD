import os, stat
from typing import List

import torch
from PIL import Image
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from huggingface_hub import hf_hub_download
from cog import BasePredictor, Path, Input


with open("concepts.txt") as infile:
    CONCEPTS = [line.rstrip() for line in infile]


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
        image: Path = Input(
            description="Inital image to generate variations of. Supproting images size with 512x512",
        ),
        mask: Path = Input(
            description="Black and white image to use as mask for inpainting over the image provided. White pixels are inpainted and black pixels are preserved",
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

        image = Image.open(image).convert("RGB").resize((512, 512))
        extra_kwargs = {
            "mask_image": Image.open(mask).convert("RGB").resize(image.size),
            "image": image
        }

        seed_everything(seed)

        repo_id_embeds = concept.split(":")[0]
        print(repo_id_embeds)
        # embeds_path = hf_hub_download(
        #     repo_id=repo_id_embeds,
        #     filename="learned_embeds.bin",
        #     cache_dir=repo_id_embeds,
        #     local_files_only=True,
        # )
        embeds_path = './metng1-5000.pt'
        # print(embeds_path)
        # token_path = hf_hub_download(
        #     repo_id=repo_id_embeds,
        #     filename="token_identifier.txt",
        #     cache_dir=repo_id_embeds,
        #     local_files_only=True,
        # )
        
        # with open(token_path, "r") as file:
        #     placeholder = file.read()

        placeholder = '<metng1>'
        print(f"The placeholder token for your concept is {placeholder}.")
        
        loaded_learned_embeds = torch.load(embeds_path, map_location="cpu")
        print(loaded_learned_embeds.keys())
        # separate token and the embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

        print(f"{trained_token}: is the trained_token.")
        print(f"{embeds}: embeds.")

        # cast to dtype of text_encoder
        dtype = self.text_encoder.get_input_embeddings().weight.dtype
        embeds.to(dtype)

        # add the token in tokenizer
        num_added_tokens = self.tokenizer.add_tokens('<metng1>')
        print(f"{num_added_tokens} new tokens added.")

        # resize the token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # get the id for the token and assign the embeds
        token_id = self.tokenizer.convert_tokens_to_ids(trained_token)
        self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

        print("loading StableDiffusionPipeline with updated tokenizer and text_encoder")
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            cache_dir="pretrain/diffusers-cache",
            local_files_only=True,
            torch_dtype=torch.float16,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        ).to("cuda")

        print("Generating images with the learned concept")
        with torch.autocast("cuda"):
            images = pipeline(
                [prompt] * num_outputs,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **extra_kwargs,
            ).images

        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
