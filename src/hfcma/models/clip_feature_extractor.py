from typing import Iterable, List

import open_clip
import torch
import torch.nn as nn


class OpenCLIPFeatureExtractor(nn.Module):
    """Frozen OpenCLIP wrapper returning global, patch, and token features."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode_image_global(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images.to(self.device), normalize=False)

    @torch.no_grad()
    def encode_text_global(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(tokens.to(self.device), normalize=False)

    @torch.no_grad()
    def encode_image_patches(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual
        x = images.to(self.device)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        if hasattr(visual, "class_embedding"):
            cls = visual.class_embedding.to(x.dtype)
            cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x) if hasattr(visual, "patch_dropout") else x
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        if x.size(1) > 1:
            x = x[:, 1:, :]
        return x

    @torch.no_grad()
    def encode_text_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        model = self.model
        x = model.token_embedding(tokens.to(self.device))
        x = x + model.positional_embedding.to(x.dtype)
        x = x.permute(1, 0, 2)
        x = model.transformer(x)
        x = x.permute(1, 0, 2)
        x = model.ln_final(x)
        mask = tokens.to(self.device).ne(0)
        return x * mask.unsqueeze(-1)

    def tokenize(self, captions: Iterable[str]) -> torch.Tensor:
        return self.tokenizer(list(captions)).to(self.device)

    def preprocess_images(self, images: List) -> torch.Tensor:
        return torch.stack([self.preprocess(image) for image in images], dim=0).to(self.device)

