# Import PyTorch main module for tensor operations and device management
import torch
# Import PyTorch neural network module for building model layers
import torch.nn as nn
# Import PyTorch functional API for activation functions and normalization ops
import torch.nn.functional as F
# Import CLIPModel from HuggingFace transformers as the vision backbone
from transformers import CLIPModel

# Import configuration constants for feature dimensions and model architecture
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


# Define a projection MLP head: two-layer linear net with LayerNorm and ReLU
class ProjectionHead(nn.Module):
    # Initialize the projection head with input, hidden, and output dimensions
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        # Call the parent nn.Module initializer
        super().__init__()
        # Build a sequential container: Linear -> LayerNorm -> ReLU -> Linear
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    # Forward pass: apply the projection network to the input tensor
    def forward(self, x):
        return self.net(x)


# Define a multimodal fusion model that combines StreetCLIP vision with object and vegetation features
class StreetCLIPFusion(nn.Module):
    # Initialize the fusion model with a pretrained CLIP backbone and configurable dimensions
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
        # Call the parent nn.Module initializer
        super().__init__()
        # Load the pretrained CLIP model from the specified checkpoint
        clip = CLIPModel.from_pretrained(streetclip_model_name)
        # Extract the vision transformer backbone
        self.vision_model = clip.vision_model
        # Extract the visual projection layer that maps to the shared embedding space
        self.visual_projection = clip.visual_projection
        # Delete the full CLIP wrapper to free memory (text tower not needed)
        del clip

        # If freeze_backbone is True, disable gradient updates for the vision encoder
        if freeze_backbone:
            # Freeze all parameters in the vision transformer backbone
            for param in self.vision_model.parameters():
                param.requires_grad = False
            # Freeze all parameters in the visual projection layer
            for param in self.visual_projection.parameters():
                param.requires_grad = False

        # Create a projection head for object-level road features (from detection models)
        self.obj_projection = ProjectionHead(obj_feature_dim, proj_hidden_dim, obj_proj_dim)
        # Create a projection head for vegetation features (from satellite/vegetation models)
        self.veg_projection = ProjectionHead(veg_feature_dim, proj_hidden_dim, veg_proj_dim)
        # Compute the total input dimension for the fusion head (CLIP + obj + veg)
        fusion_input_dim = STREETCLIP_EMBED_DIM + obj_proj_dim + veg_proj_dim
        # Build the fusion MLP head: 3-layer net with LayerNorm, ReLU, and Dropout
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

    # Encode an image through the frozen vision backbone and projection
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Pass pixel values through the vision transformer
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        # Extract the pooled output representation (CLIP-style pooling)
        pooled = vision_outputs.pooler_output
        # Project the pooled representation into the shared embedding space
        return self.visual_projection(pooled)

    # Forward pass: fuse visual, object, and vegetation features into a single embedding
    def forward(
        self,
        pixel_values: torch.Tensor = None,
        road_features: torch.Tensor = None,
        veg_features: torch.Tensor = None,
        embeddings: torch.Tensor = None,
    ):
        # If precomputed embeddings are provided, use them directly
        if embeddings is not None:
            streetclip_emb = embeddings
        # Otherwise, if pixel values are provided, encode them through the vision model
        elif pixel_values is not None:
            streetclip_emb = self.encode_image(pixel_values)
        # If neither embeddings nor pixel values are given, raise an error
        else:
            raise ValueError("Either pixel_values or embeddings must be provided")

        # Project object features into the common embedding dimension
        obj_proj = self.obj_projection(road_features)
        # Project vegetation features into the common embedding dimension
        veg_proj = self.veg_projection(veg_features)
        # Concatenate the three feature vectors along the last dimension
        fused = torch.cat([streetclip_emb, obj_proj, veg_proj], dim=-1)
        # Pass the concatenated features through the fusion MLP head
        embedding = self.fusion_head(fused)
        # L2-normalize the final embedding to unit length
        embedding = F.normalize(embedding, p=2, dim=-1)
        # Return the normalized fusion embedding
        return embedding


# Define a supervised contrastive loss for learning discriminative embeddings
class ContrastiveLoss(nn.Module):
    # Initialize the contrastive loss with a temperature scaling parameter
    def __init__(self, temperature: float = 0.07):
        # Call the parent nn.Module initializer
        super().__init__()
        # Store the temperature parameter for scaling logits
        self.temperature = temperature

    # Compute the supervised contrastive loss given embeddings and labels
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        # Get the device of the input embeddings for consistent tensor placement
        device = embeddings.device
        # Get the batch size from the first dimension of the embeddings
        batch_size = embeddings.shape[0]
        # Compute the cosine similarity matrix scaled by temperature (logits)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        # Create a binary mask where mask[i][j]=1 if labels[i]==labels[j], else 0
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(device)
        # Set the diagonal of the mask to 0 (exclude self from positive set)
        mask.fill_diagonal_(0)
        # Compute exponentiated similarities, excluding self (diagonal set to 1 * 0)
        exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(batch_size, device=device))
        # Sum of exponentiated positive similarities for each anchor
        pos_sum = (exp_sim * mask).sum(dim=1)
        # Sum of all exponentiated similarities for each anchor (denominator)
        all_sum = exp_sim.sum(dim=1)
        # Compute per-sample contrastive loss with epsilon for numerical stability
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        # Identify which samples have at least one positive pair
        valid = mask.sum(dim=1) > 0
        # If any samples have valid positive pairs, average only over those
        if valid.sum() > 0:
            loss = loss[valid].mean()
        # Otherwise, average over all samples (edge case with no positive pairs)
        else:
            loss = loss.mean()
        # Return the scalar contrastive loss value
        return loss


# Define a multimodal contrastive loss aligning image and text embeddings
class MultiModalContrastiveLoss(nn.Module):
    # Initialize with temperature, text input dim, and output embedding dim
    def __init__(self, temperature: float = 0.07, text_dim: int = STREETCLIP_EMBED_DIM, output_dim: int = FUSION_OUTPUT_DIM):
        # Call the parent nn.Module initializer
        super().__init__()
        # Store the temperature parameter
        self.temperature = temperature
        # Learnable log-scaled temperature parameter initialized from the temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        # Project text embeddings to match the output embedding dimension
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    # Forward pass: compute symmetric cross-entropy loss between image and text embeddings
    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        # Get the device from the image embeddings tensor
        device = image_embeddings.device
        # L2-normalize image embeddings to unit length
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        # Project text embeddings into the output space
        text_embeddings = self.text_projection(text_embeddings)
        # L2-normalize projected text embeddings to unit length
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        # Exponentiate the log scale to obtain the actual temperature scale factor
        logit_scale = self.logit_scale.exp()
        # Compute scaled cosine similarity logits between all image-text pairs
        logits = logit_scale * torch.matmul(image_embeddings, text_embeddings.T)
        # Get the batch size from the image embeddings
        batch_size = image_embeddings.shape[0]
        # Create target indices representing the correct image-text pairs (diagonal)
        target = torch.arange(batch_size, device=device)
        # Compute cross-entropy loss treating images as anchors
        loss_i = F.cross_entropy(logits, target)
        # Compute cross-entropy loss treating texts as anchors (transpose logits)
        loss_t = F.cross_entropy(logits.T, target)
        # Return the average of the two symmetric losses
        return (loss_i + loss_t) / 2
