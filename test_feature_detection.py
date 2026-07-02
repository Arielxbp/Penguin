import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from config import (
    AUGMENTATIONS_PER_IMAGE,
    GROUNDING_DINO_MODEL,
    ROAD_CATEGORIES,
    VEGETATION_CATEGORIES,
    YOLO_WORLD_MODEL,
)
from dataset import PIL_AUG_TRANSFORM, TENSOR_AUG_TRANSFORM

STREETCLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
STREETCLIP_STD = [0.26862954, 0.26130258, 0.27577711]

BOX_COLORS = [
    "#FF4444", "#44FF44", "#4488FF", "#FFAA00", "#FF44FF",
    "#44FFFF", "#FFFF44", "#FF8844", "#88FF44", "#44FF88",
    "#8844FF", "#FF4488", "#FF8888", "#88FF88", "#8888FF",
]


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.new_tensor(STREETCLIP_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(STREETCLIP_STD).view(3, 1, 1)
    return tensor * std + mean


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    t = denormalize(tensor).clamp(0, 1).cpu()
    return T.ToPILImage()(t)


def generate_augmented_variants(image: Image.Image, n_variants: int):
    variants = []
    for _ in range(n_variants):
        aug_img = PIL_AUG_TRANSFORM(image)
        aug_tensor = T.ToTensor()(aug_img)
        aug_tensor = TENSOR_AUG_TRANSFORM(aug_tensor)
        aug_tensor = T.Normalize(mean=STREETCLIP_MEAN, std=STREETCLIP_STD)(aug_tensor)
        variants.append(aug_tensor)
    return variants


class GroundingDINODetector:
    def __init__(self, device: str, box_threshold: float = 0.3, text_threshold: float = 0.25):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.categories = ROAD_CATEGORIES
        self._processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL).to(device)
        self._model.eval()
        self._clip_mean = torch.tensor(STREETCLIP_MEAN).view(3, 1, 1)
        self._clip_std = torch.tensor(STREETCLIP_STD).view(3, 1, 1)

    def _to_pil(self, image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        t = image.cpu() * self._clip_std + self._clip_mean
        t = torch.clamp(t, 0, 1)
        return T.ToPILImage()(t)

    @torch.inference_mode()
    def detect(self, image):
        pil = self._to_pil(image)
        text_query = ". ".join([f"a {cat}" for cat in self.categories]) + "."
        inputs = self._processor(images=pil, text=text_query, return_tensors="pt").to(self.device)
        outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil.size[::-1]],
        )
        result = results[0] if results else None

        detections = []
        if result and len(result) > 0:
            boxes = result.get("boxes", [])
            labels = result.get("labels", [])
            scores = result.get("scores", [])
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.tolist()
                detections.append((label, float(score), (x1, y1, x2, y2)))

        feature_vec = self._build_feature_vec(detections)
        return feature_vec, detections

    def _build_feature_vec(self, detections: list) -> torch.Tensor:
        feature_vec = torch.zeros(len(self.categories))
        category_scores = {}
        for label, score, _ in detections:
            label_lower = label.lower()
            for i, cat in enumerate(self.categories):
                if cat in label_lower or label_lower in cat:
                    category_scores[i] = max(category_scores.get(i, 0), score)
        for idx, score in category_scores.items():
            feature_vec[idx] = score
        return feature_vec


class YOLOWorldDetector:
    def __init__(self, device: str, conf_threshold: float = 0.25):
        try:
            from ultralytics import YOLOWorld
        except ImportError:
            raise RuntimeError("ultralytics not installed; YOLOWorld unavailable")
        self.device = device
        self.conf_threshold = conf_threshold
        self.categories = ROAD_CATEGORIES
        self._clip_mean = torch.tensor(STREETCLIP_MEAN).view(3, 1, 1)
        self._clip_std = torch.tensor(STREETCLIP_STD).view(3, 1, 1)
        self._model = YOLOWorld(YOLO_WORLD_MODEL).to(device)
        self._model.set_classes(self.categories)
        self._model.eval()

    def _to_pil(self, image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        t = image.cpu() * self._clip_std + self._clip_mean
        t = torch.clamp(t, 0, 1)
        return T.ToPILImage()(t)

    @torch.inference_mode()
    def detect(self, image):
        pil = self._to_pil(image)
        results = self._model.predict(pil, conf=self.conf_threshold, verbose=False)

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                class_ids = boxes.cls.cpu().int().numpy()
                confidences = boxes.conf.cpu().numpy()
                coords = boxes.xyxy.cpu().numpy()
                for cls_id, conf, xyxy in zip(class_ids, confidences, coords):
                    if 0 <= cls_id < len(self.categories):
                        label = self.categories[int(cls_id)]
                        x1, y1, x2, y2 = xyxy.tolist()
                        detections.append((label, float(conf), (x1, y1, x2, y2)))

        feature_vec = self._build_feature_vec(detections)
        return feature_vec, detections

    def _build_feature_vec(self, detections: list) -> torch.Tensor:
        feature_vec = torch.zeros(len(self.categories))
        category_scores = {}
        for label, score, _ in detections:
            for i, cat in enumerate(self.categories):
                if cat in label.lower() or label.lower() in cat:
                    category_scores[i] = max(category_scores.get(i, 0), score)
        for idx, score in category_scores.items():
            feature_vec[idx] = score
        return feature_vec


def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i, (label, score, box) in enumerate(detections):
        x1, y1, x2, y2 = box
        color = BOX_COLORS[i % len(BOX_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{label} ({score:.2f})"
        bbox = draw.textbbox((x1, y1 - 16), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 16), text, fill="white", font=font)
    return img


def print_features(label, feature_vec, categories, top_k=10):
    scores = feature_vec.cpu().numpy()
    nonzero = [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0]
    nonzero.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{label} ({len(nonzero)} categories detected):")
    print("-" * 60)
    for idx, score in nonzero[:top_k]:
        print(f"  {categories[idx]:<30s}  score={score:.4f}")
    if len(nonzero) > top_k:
        print(f"  ... and {len(nonzero) - top_k} more")


def main():
    parser = argparse.ArgumentParser(
        description="Run feature detection on a single photo, save augmented images and draw bounding boxes."
    )
    parser.add_argument("image", type=str, help="Path to the input photo")
    parser.add_argument(
        "--road-model",
        choices=["grounding_dino", "yolo_world"],
        default="grounding_dino",
    )
    parser.add_argument(
        "--veg-model",
        choices=["clip", "ram++"],
        default="clip",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "-n",
        type=int,
        default=AUGMENTATIONS_PER_IMAGE,
        help="Number of augmented variants to test (default: AUGMENTATIONS_PER_IMAGE)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top-scoring categories to display",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="output/feature_test",
        help="Output directory for images",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.3,
        help="GroundingDINO box threshold",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="GroundingDINO text threshold",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="YOLOWorld confidence threshold",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: image not found: {args.image}")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    from features import CLIPBasedVegetationExtractor, RAMPlusExtractor
    if args.veg_model == "clip":
        veg_extractor = CLIPBasedVegetationExtractor(device=args.device)
    else:
        veg_extractor = RAMPlusExtractor(device=args.device)

    print(f"Loading road detector: {args.road_model}")
    if args.road_model == "grounding_dino":
        road_detector = GroundingDINODetector(
            device=args.device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
    else:
        road_detector = YOLOWorldDetector(
            device=args.device,
            conf_threshold=args.conf_threshold,
        )
    print(f"  road feature dim: {len(ROAD_CATEGORIES)}")
    print(f"  veg feature dim:  {veg_extractor.feature_dim()}")

    image = Image.open(img_path).convert("RGB")
    print(f"\nImage: {img_path}  ({image.size[0]}x{image.size[1]})")

    print("\n" + "=" * 60)
    print("ORIGINAL IMAGE")
    print("=" * 60)

    road_feat, road_dets = road_detector.detect(image)
    veg_feat = veg_extractor.extract(image)

    print_features("Road features", road_feat, ROAD_CATEGORIES, top_k=args.top_k)
    print_features("Vegetation features", veg_feat, VEGETATION_CATEGORIES, top_k=args.top_k)

    print(f"\nBounding boxes found: {len(road_dets)}")
    for label, score, box in road_dets:
        print(f"  {label:<30s} score={score:.4f}  box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})")

    if road_dets:
        boxed = draw_boxes(image, road_dets)
        boxed_path = outdir / f"{stem}_boxes.png"
        boxed.save(boxed_path)
        print(f"Saved annotated image: {boxed_path}")

    all_road_feats = [road_feat]
    all_veg_feats = [veg_feat]

    if args.n > 0:
        print(f"\nGenerating {args.n} augmented variant(s)...")
        variants = generate_augmented_variants(image, args.n)

        for j, variant_tensor in enumerate(variants):
            aug_pil = tensor_to_pil(variant_tensor)
            aug_path = outdir / f"{stem}_aug{j}.png"
            aug_pil.save(aug_path)

            aug_road_feat, aug_road_dets = road_detector.detect(variant_tensor)
            aug_veg_feat = veg_extractor.extract(variant_tensor)

            all_road_feats.append(aug_road_feat)
            all_veg_feats.append(aug_veg_feat)

            print(f"\n{'=' * 60}")
            print(f"AUGMENTED VARIANT {j + 1}")
            print(f"{'=' * 60}")
            print(f"Saved: {aug_path}")
            print(f"Bounding boxes found: {len(aug_road_dets)}")
            for label, score, box in aug_road_dets:
                print(f"  {label:<30s} score={score:.4f}  box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})")
            print_features(
                f"Road features (aug {j + 1})",
                aug_road_feat,
                ROAD_CATEGORIES,
                top_k=args.top_k,
            )
            print_features(
                f"Vegetation features (aug {j + 1})",
                aug_veg_feat,
                VEGETATION_CATEGORIES,
                top_k=args.top_k,
            )

            if aug_road_dets:
                aug_boxed = draw_boxes(aug_pil, aug_road_dets)
                aug_boxed_path = outdir / f"{stem}_aug{j}_boxes.png"
                aug_boxed.save(aug_boxed_path)
                print(f"Saved annotated image: {aug_boxed_path}")

        mean_road = torch.stack(all_road_feats).mean(dim=0)
        mean_veg = torch.stack(all_veg_feats).mean(dim=0)
        print(f"\n{'=' * 60}")
        print("MEAN OVER ORIGINAL + AUGMENTED")
        print(f"{'=' * 60}")
        print_features("Road features (mean)", mean_road, ROAD_CATEGORIES, top_k=args.top_k)
        print_features("Vegetation features (mean)", mean_veg, VEGETATION_CATEGORIES, top_k=args.top_k)

    print(f"\nDone. Output in {outdir}/")


if __name__ == "__main__":
    main()
