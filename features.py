from abc import ABC, abstractmethod
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from huggingface_hub import hf_hub_download

from config import (
    GROUNDING_DINO_MODEL,
    OBJ_FEATURE_DIM,
    RAM_PLUS_MODEL,
    ROAD_CATEGORIES,
    VEG_FEATURE_DIM,
    VEGETATION_CATEGORIES,
    YOLO_WORLD_MODEL,
)


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image: Image.Image) -> torch.Tensor:
        pass

    def extract_batch(self, images: list) -> torch.Tensor:
        return torch.stack([self.extract(img) for img in images])

    @abstractmethod
    def feature_dim(self) -> int:
        pass


class GroundingDINOExtractor(FeatureExtractor):
    def __init__(self, device: str = "cuda", box_threshold: float = 0.3, text_threshold: float = 0.25,
                 max_batch_size: int = 2):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_batch_size = max_batch_size
        self.categories = ROAD_CATEGORIES
        self._model = None
        self._processor = None
        self._clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self._clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def _load(self):
        if self._model is None:
            self._processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                GROUNDING_DINO_MODEL
            ).to(self.device)
            self._model.eval()

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.cpu() * self._clip_std + self._clip_mean
        tensor = torch.clamp(tensor, 0, 1)
        return T.ToPILImage()(tensor)

    def feature_dim(self) -> int:
        return OBJ_FEATURE_DIM

    def _build_feature_vec(self, result) -> torch.Tensor:
        feature_vec = torch.zeros(len(self.categories))
        if result and len(result) > 0:
            labels = result.get("labels", [])
            scores = result.get("scores", [])
            category_scores = {}
            for label, score in zip(labels, scores):
                label_lower = label.lower()
                for i, cat in enumerate(self.categories):
                    if cat in label_lower or label_lower in cat:
                        category_scores[i] = max(category_scores.get(i, 0), score)
            for idx, score in category_scores.items():
                feature_vec[idx] = score
        return feature_vec

    @torch.inference_mode()
    def extract(self, image) -> torch.Tensor:
        self._load()
        if not isinstance(image, Image.Image):
            image = self._tensor_to_pil(image)
        text_query = ". ".join([f"a {cat}" for cat in self.categories]) + "."
        inputs = self._processor(images=image, text=text_query, return_tensors="pt").to(self.device)
        outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )
        return self._build_feature_vec(results[0] if results else None)

    @torch.inference_mode()
    def extract_batch(self, images: list) -> torch.Tensor:
        self._load()
        text_query = ". ".join([f"a {cat}" for cat in self.categories]) + "."
        pil_images = []
        target_sizes = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = self._tensor_to_pil(img)
            pil_images.append(img)
            target_sizes.append(img.size[::-1])
        all_results = []
        for start in range(0, len(pil_images), self.max_batch_size):
            end = start + self.max_batch_size
            sub_images = pil_images[start:end]
            sub_sizes = target_sizes[start:end]
            texts = [text_query] * len(sub_images)
            inputs = self._processor(images=sub_images, text=texts, return_tensors="pt").to(self.device)
            outputs = self._model(**inputs)
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=sub_sizes,
            )
            all_results.extend(results)
        return torch.stack([self._build_feature_vec(r) for r in all_results])


class YOLOWorldExtractor(FeatureExtractor):
    def __init__(self, device: str = "cuda", conf_threshold: float = 0.25):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.conf_threshold = conf_threshold
        self.categories = ROAD_CATEGORIES
        self._model = None
        self._clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self._clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        try:
            from ultralytics import YOLOWorld
            self._model_cls = YOLOWorld
        except ImportError:
            self._model_cls = None

    def _load(self):
        if self._model is None and self._model_cls is not None:
            self._model = self._model_cls(YOLO_WORLD_MODEL)
            self._model.to(self.device)
            self._model.set_classes(self.categories)
            self._model.eval()

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.cpu() * self._clip_std + self._clip_mean
        tensor = torch.clamp(tensor, 0, 1)
        return T.ToPILImage()(tensor)

    def feature_dim(self) -> int:
        return OBJ_FEATURE_DIM

    @torch.inference_mode()
    def extract(self, image) -> torch.Tensor:
        if self._model_cls is None:
            return torch.zeros(self.feature_dim())
        self._load()
        if not isinstance(image, Image.Image):
            image = self._tensor_to_pil(image)
        results = self._model.predict(image, conf=self.conf_threshold, verbose=False)
        feature_vec = torch.zeros(len(self.categories))
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                class_ids = boxes.cls.cpu().int().numpy()
                confidences = boxes.conf.cpu().numpy()
                category_scores = {}
                for cls_id, conf in zip(class_ids, confidences):
                    idx = int(cls_id)
                    if 0 <= idx < len(self.categories):
                        category_scores[idx] = max(category_scores.get(idx, 0), float(conf))
                for idx, score in category_scores.items():
                    feature_vec[idx] = score
        return feature_vec


class RAMPlusExtractor(FeatureExtractor):
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.veg_categories = VEGETATION_CATEGORIES
        self._model = None
        self._transform = None

    def _load(self):
        if self._model is not None:
            return
        from recognize_anything.inference import load_ram_plus

        model_path = hf_hub_download(
            repo_id=RAM_PLUS_MODEL,
            filename="ram_plus_swin_large_14m.pth",
        )
        embed_path = hf_hub_download(
            repo_id=RAM_PLUS_MODEL,
            filename="ram_plus_tag_embedding_class_4585_des_51.pth",
        )
        self._model, self._transform = load_ram_plus(
            model_path,
            embed_path,
            device=self.device,
        )
        self._model.eval()

    def feature_dim(self) -> int:
        return VEG_FEATURE_DIM

    @torch.inference_mode()
    def extract(self, image: Image.Image) -> torch.Tensor:
        self._load()
        result = self._model.inference(image, self._transform)
        tags = result.get("tags", [])
        tag_lower = [t.lower() for t in tags]
        feature_vec = torch.zeros(len(self.veg_categories))
        for i, veg_cat in enumerate(self.veg_categories):
            for tag in tag_lower:
                if veg_cat in tag or tag in veg_cat:
                    feature_vec[i] = 1.0
                    break
        return feature_vec


class CLIPBasedVegetationExtractor(FeatureExtractor):
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.veg_categories = VEGETATION_CATEGORIES
        self._model = None
        self._processor = None
        self._clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self._clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def _load(self):
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self._model.eval()

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        tensor = tensor.cpu() * self._clip_std + self._clip_mean
        tensor = torch.clamp(tensor, 0, 1)
        return T.ToPILImage()(tensor)

    def feature_dim(self) -> int:
        return VEG_FEATURE_DIM

    @torch.inference_mode()
    def extract(self, image) -> torch.Tensor:
        self._load()
        if not isinstance(image, Image.Image):
            image = self._tensor_to_pil(image)
        texts = [f"a photo of {cat}" for cat in self.veg_categories]
        inputs = self._processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self._model(**inputs)
        logits_per_image = outputs.logits_per_image[0]
        probs = logits_per_image.softmax(dim=0)
        return probs.cpu()

    @torch.inference_mode()
    def extract_batch(self, images: list) -> torch.Tensor:
        self._load()
        pil_images = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = self._tensor_to_pil(img)
            pil_images.append(img)
        texts = [f"a photo of {cat}" for cat in self.veg_categories]
        inputs = self._processor(text=texts, images=pil_images, return_tensors="pt", padding=True).to(self.device)
        outputs = self._model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)
        return probs.cpu()


class CompositeFeatureExtractor:
    def __init__(
        self,
        road_extractor: FeatureExtractor,
        veg_extractor: FeatureExtractor,
        device: str = "cuda",
    ):
        self.road_extractor = road_extractor
        self.veg_extractor = veg_extractor
        self.device = device if torch.cuda.is_available() else "cpu"

    @torch.inference_mode()
    def extract(self, image: Image.Image):
        road_features = self.road_extractor.extract(image)
        veg_features = self.veg_extractor.extract(image)
        return {
            "road_features": road_features,
            "veg_features": veg_features,
        }

    @torch.inference_mode()
    def extract_batch(self, images: list):
        road_features = self.road_extractor.extract_batch(images)
        veg_features = self.veg_extractor.extract_batch(images)
        return {
            "road_features": road_features,
            "veg_features": veg_features,
        }

    def road_dim(self) -> int:
        return self.road_extractor.feature_dim()

    def veg_dim(self) -> int:
        return self.veg_extractor.feature_dim()


def create_feature_extractors(
    road_model: str = "grounding_dino",
    veg_model: str = "clip",
    device: str = "cuda",
):
    device = device if torch.cuda.is_available() else "cpu"
    if road_model == "grounding_dino":
        road_extractor = GroundingDINOExtractor(device=device)
    elif road_model == "yolo_world":
        road_extractor = YOLOWorldExtractor(device=device)
    else:
        raise ValueError(f"Unknown road model: {road_model}")

    if veg_model == "ram++":
        veg_extractor = RAMPlusExtractor(device=device)
    elif veg_model == "clip":
        veg_extractor = CLIPBasedVegetationExtractor(device=device)
    else:
        raise ValueError(f"Unknown veg model: {veg_model}")

    return CompositeFeatureExtractor(road_extractor, veg_extractor, device=device)
