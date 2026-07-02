import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from config import (
    OBJ_FEATURE_DIM,
    OBJ_PROJECTION_DIM,
    PROJECTION_HIDDEN_DIM,
    FUSION_OUTPUT_DIM,
    STREETCLIP_EMBED_DIM,
    STREETCLIP_MODEL,
    VEG_FEATURE_DIM,
    VEG_PROJECTION_DIM,
)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class StreetCLIPFusion(nn.Module):
    def __init__(
        self,
        streetclip_model_name: str = STREETCLIP_MODEL,
        obj_feature_dim: int = OBJ_FEATURE_DIM,
        veg_feature_dim: int = VEG_FEATURE_DIM,
        obj_proj_dim: int = OBJ_PROJECTION_DIM,
        veg_proj_dim: int = VEG_PROJECTION_DIM,
        proj_hidden_dim: int = PROJECTION_HIDDEN_DIM,
        fusion_output_dim: int = FUSION_OUTPUT_DIM,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        clip = CLIPModel.from_pretrained(streetclip_model_name)
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection
        del clip

        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in self.visual_projection.parameters():
                param.requires_grad = False

        self.obj_projection = ProjectionHead(obj_feature_dim, proj_hidden_dim, obj_proj_dim)
        self.veg_projection = ProjectionHead(veg_feature_dim, proj_hidden_dim, veg_proj_dim)
        fusion_input_dim = STREETCLIP_EMBED_DIM + obj_proj_dim + veg_proj_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, proj_hidden_dim * 2),
            nn.LayerNorm(proj_hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(proj_hidden_dim * 2, proj_hidden_dim),
            nn.LayerNorm(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(proj_hidden_dim, fusion_output_dim),
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        return self.visual_projection(pooled)

    def forward(
        self,
        pixel_values: torch.Tensor = None,
        road_features: torch.Tensor = None,
        veg_features: torch.Tensor = None,
        embeddings: torch.Tensor = None,
    ):
        if embeddings is not None:
            streetclip_emb = embeddings
        elif pixel_values is not None:
            streetclip_emb = self.encode_image(pixel_values)
        else:
            raise ValueError("Either pixel_values or embeddings must be provided")

        obj_proj = self.obj_projection(road_features)
        veg_proj = self.veg_projection(veg_features)
        fused = torch.cat([streetclip_emb, obj_proj, veg_proj], dim=-1)
        embedding = self.fusion_head(fused)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        device = embeddings.device
        batch_size = embeddings.shape[0]
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(device)
        mask.fill_diagonal_(0)
        exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(batch_size, device=device))
        pos_sum = (exp_sim * mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        valid = mask.sum(dim=1) > 0
        if valid.sum() > 0:
            loss = loss[valid].mean()
        else:
            loss = loss.mean()
        return loss


class MultiModalContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, text_dim: int = STREETCLIP_EMBED_DIM, output_dim: int = FUSION_OUTPUT_DIM):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        device = image_embeddings.device
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = self.text_projection(text_embeddings)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(image_embeddings, text_embeddings.T)
        batch_size = image_embeddings.shape[0]
        target = torch.arange(batch_size, device=device)
        loss_i = F.cross_entropy(logits, target)
        loss_t = F.cross_entropy(logits.T, target)
        return (loss_i + loss_t) / 2
