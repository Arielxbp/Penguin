# Import argparse for parsing command-line arguments in the main function.
import argparse
# Import Path from pathlib for cross-platform filesystem path handling.
from pathlib import Path

# Import PyTorch for tensor operations and GPU/CPU device management.
import torch
# Import torchvision transforms for image conversions and normalization.
import torchvision.transforms as T
# Import PIL Image for loading and manipulating image files, and ImageDraw/ImageFont for annotation.
from PIL import Image, ImageDraw, ImageFont

# Import configuration constants: augmentation count, model identifiers, and category lists.
from config import (
    AUGMENTATIONS_PER_IMAGE,
    GROUNDING_DINO_MODEL,
    ROAD_CATEGORIES,
    VEGETATION_CATEGORIES,
    YOLO_WORLD_MODEL,
)
# Import augmentation transforms: one for PIL images and another for tensor-based augmentations.
from dataset import PIL_AUG_TRANSFORM, TENSOR_AUG_TRANSFORM

# Define the mean values (per channel) used by the StreetCLIP normalization.
STREETCLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
# Define the standard deviation values (per channel) used by the StreetCLIP normalization.
STREETCLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Define a palette of 15 hex color strings used to draw bounding boxes for detections.
BOX_COLORS = [
    "#FF4444", "#44FF44", "#4488FF", "#FFAA00", "#FF44FF",
    "#44FFFF", "#FFFF44", "#FF8844", "#88FF44", "#44FF88",
    "#8844FF", "#FF4488", "#FF8888", "#88FF88", "#8888FF",
]


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    # Create a tensor of mean values reshaped to (3, 1, 1) on the same device as the input tensor.
    mean = tensor.new_tensor(STREETCLIP_MEAN).view(3, 1, 1)
    # Create a tensor of standard deviation values reshaped to (3, 1, 1) on the same device.
    std = tensor.new_tensor(STREETCLIP_STD).view(3, 1, 1)
    # Reverse the normalization: multiply by std and add mean to recover the original pixel range.
    return tensor * std + mean


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    # Denormalize the tensor and clamp values to the [0, 1] range, then move to CPU.
    t = denormalize(tensor).clamp(0, 1).cpu()
    # Convert the (3, H, W) tensor to a PIL Image in (H, W, 3) uint8 format.
    return T.ToPILImage()(t)


def generate_augmented_variants(image: Image.Image, n_variants: int):
    # Initialize an empty list to accumulate augmented image tensors.
    variants = []
    # Generate the requested number of augmented variants in a loop.
    for _ in range(n_variants):
        # Apply the PIL-level augmentation transforms (e.g., color jitter, flip) to the input image.
        aug_img = PIL_AUG_TRANSFORM(image)
        # Convert the augmented PIL image to a (C, H, W) float tensor in [0, 1].
        aug_tensor = T.ToTensor()(aug_img)
        # Apply tensor-level augmentations (e.g., additional noise or blur).
        aug_tensor = TENSOR_AUG_TRANSFORM(aug_tensor)
        # Normalize the augmented tensor using StreetCLIP mean and std.
        aug_tensor = T.Normalize(mean=STREETCLIP_MEAN, std=STREETCLIP_STD)(aug_tensor)
        # Append the fully processed augmented tensor to the variants list.
        variants.append(aug_tensor)
    # Return the list of augmented image tensors.
    return variants


class GroundingDINODetector:
    # Box and text thresholds are passed to the grounding DINO post-processor to filter detections.
    def __init__(self, device: str, box_threshold: float = 0.3, text_threshold: float = 0.25):
        # Import the HuggingFace Transformers classes needed for Grounding DINO at initialization time.
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        # Store the target compute device (e.g., "cuda" or "cpu").
        self.device = device
        # Store the minimum box confidence threshold for filtering detected boxes.
        self.box_threshold = box_threshold
        # Store the minimum text alignment threshold for filtering label-box associations.
        self.text_threshold = text_threshold
        # Store the list of road categories (used to build the text query and feature vector).
        self.categories = ROAD_CATEGORIES
        # Load the pre-trained Grounding DINO processor for image/text tokenization and post-processing.
        self._processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
        # Load the pre-trained Grounding DINO zero-shot object detection model and move it to the target device.
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL).to(device)
        # Set the model to evaluation mode to disable dropout and other training-specific behavior.
        self._model.eval()
        # Create a tensor of CLIP/StreetCLIP mean values reshaped to (3, 1, 1) for denormalization.
        self._clip_mean = torch.tensor(STREETCLIP_MEAN).view(3, 1, 1)
        # Create a tensor of CLIP/StreetCLIP std values reshaped to (3, 1, 1) for denormalization.
        self._clip_std = torch.tensor(STREETCLIP_STD).view(3, 1, 1)

    def _to_pil(self, image) -> Image.Image:
        # If the input is already a PIL Image, return it as-is.
        if isinstance(image, Image.Image):
            return image
        # Denormalize the tensor back to [0, 1] using the stored CLIP mean and std.
        t = image.cpu() * self._clip_std + self._clip_mean
        # Clamp pixel values to the valid [0, 1] range.
        t = torch.clamp(t, 0, 1)
        # Convert the denormalized (3, H, W) tensor back to a PIL Image.
        return T.ToPILImage()(t)

    # Use PyTorch inference mode to disable gradient computation for faster detection.
    @torch.inference_mode()
    def detect(self, image):
        # Convert the input (PIL or normalized tensor) to a PIL Image for the processor.
        pil = self._to_pil(image)
        # Build a natural-language text query by joining all road categories with ". " separators.
        text_query = ". ".join([f"a {cat}" for cat in self.categories]) + "."
        # Tokenize the image and text query, return PyTorch tensors, and move them to the target device.
        inputs = self._processor(images=pil, text=text_query, return_tensors="pt").to(self.device)
        # Run the model forward pass to produce detection outputs.
        outputs = self._model(**inputs)
        # Post-process the raw model outputs into bounding boxes, labels, and scores.
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil.size[::-1]],
        )
        # Extract the first (and typically only) result from the post-processed list.
        result = results[0] if results else None

        # Build a list of (label, score, box) tuples from the detection result.
        detections = []
        # Only process results if we have a non-empty result dictionary.
        if result and len(result) > 0:
            # Extract the list of bounding boxes in [x1, y1, x2, y2] format.
            boxes = result.get("boxes", [])
            # Extract the list of textual label strings for each detected box.
            labels = result.get("labels", [])
            # Extract the list of confidence scores for each detection.
            scores = result.get("scores", [])
            # Iterate over boxes, labels, and scores together.
            for box, label, score in zip(boxes, labels, scores):
                # Convert the box tensor to a Python list of four coordinates.
                x1, y1, x2, y2 = box.tolist()
                # Append the detection as a (label, score, box_tuple) entry.
                detections.append((label, float(score), (x1, y1, x2, y2)))

        # Build a fixed-length feature vector by mapping detected labels to category indices.
        feature_vec = self._build_feature_vec(detections)
        # Return both the feature vector and the raw detection list.
        return feature_vec, detections

    def _build_feature_vec(self, detections: list) -> torch.Tensor:
        # Initialize a zero-filled feature vector with one element per road category.
        feature_vec = torch.zeros(len(self.categories))
        # Dictionary to accumulate the maximum score for each category index.
        category_scores = {}
        # Iterate over each detection (label, score, box).
        for label, score, _ in detections:
            # Normalize the label to lowercase for case-insensitive matching.
            label_lower = label.lower()
            # Check each category to see if it appears in the detection label (or vice versa).
            for i, cat in enumerate(self.categories):
                # If the category string is a substring of the label or the label is a substring of the category.
                if cat in label_lower or label_lower in cat:
                    # Store the maximum score seen for this category index.
                    category_scores[i] = max(category_scores.get(i, 0), score)
        # Write the accumulated scores into the corresponding positions in the feature vector.
        for idx, score in category_scores.items():
            # Set the feature vector value at index idx to the highest score for that category.
            feature_vec[idx] = score
        # Return the filled feature vector.
        return feature_vec


class YOLOWorldDetector:
    # Confidence threshold filters out low-confidence detections from YOLO-World predictions.
    def __init__(self, device: str, conf_threshold: float = 0.25):
        # Attempt to import the YOLOWorld class from ultralytics; raise an error if not installed.
        try:
            from ultralytics import YOLOWorld
        except ImportError:
            raise RuntimeError("ultralytics not installed; YOLOWorld unavailable")
        # Store the target compute device (e.g., "cuda" or "cpu").
        self.device = device
        # Store the minimum confidence threshold for filtering detections.
        self.conf_threshold = conf_threshold
        # Store the list of road categories for the text prompt and feature vector.
        self.categories = ROAD_CATEGORIES
        # Create a tensor of CLIP/StreetCLIP mean values reshaped to (3, 1, 1) for denormalization.
        self._clip_mean = torch.tensor(STREETCLIP_MEAN).view(3, 1, 1)
        # Create a tensor of CLIP/StreetCLIP std values reshaped to (3, 1, 1) for denormalization.
        self._clip_std = torch.tensor(STREETCLIP_STD).view(3, 1, 1)
        # Load the YOLO-World model from the pre-trained weights identifier and move to the target device.
        self._model = YOLOWorld(YOLO_WORLD_MODEL).to(device)
        # Set the text prompt classes of the YOLO-World model to our road categories.
        self._model.set_classes(self.categories)
        # Set the model to evaluation mode to disable training-specific layers.
        self._model.eval()

    def _to_pil(self, image) -> Image.Image:
        # If the input is already a PIL Image, return it without modification.
        if isinstance(image, Image.Image):
            return image
        # Denormalize the tensor back to [0, 1] pixel range using the stored CLIP mean and std.
        t = image.cpu() * self._clip_std + self._clip_mean
        # Clamp all values to the valid [0, 1] range to avoid out-of-bounds pixel values.
        t = torch.clamp(t, 0, 1)
        # Convert the (3, H, W) tensor back to a PIL Image.
        return T.ToPILImage()(t)

    # Use PyTorch inference mode to disable gradient tracking for faster detection.
    @torch.inference_mode()
    def detect(self, image):
        # Convert the input (PIL or normalized tensor) to a PIL Image for the YOLO model.
        pil = self._to_pil(image)
        # Run YOLO-World prediction on the image with the configured confidence threshold and no verbose logging.
        results = self._model.predict(pil, conf=self.conf_threshold, verbose=False)

        # Accumulate (label, score, box) tuples from the YOLO prediction results.
        detections = []
        # Check that the results list is non-empty.
        if results and len(results) > 0:
            # Extract the bounding boxes container from the first (only) result.
            boxes = results[0].boxes
            # Only process if there are actual detected boxes.
            if boxes is not None and len(boxes) > 0:
                # Get the integer class IDs for each detected box, moved to CPU.
                class_ids = boxes.cls.cpu().int().numpy()
                # Get the confidence scores for each detected box, moved to CPU.
                confidences = boxes.conf.cpu().numpy()
                # Get the bounding box coordinates in [x1, y1, x2, y2] format, moved to CPU.
                coords = boxes.xyxy.cpu().numpy()
                # Iterate over class IDs, confidence scores, and box coordinates together.
                for cls_id, conf, xyxy in zip(class_ids, confidences, coords):
                    # Ensure the class ID is within the valid range of our categories list.
                    if 0 <= cls_id < len(self.categories):
                        # Look up the human-readable label string from the categories list.
                        label = self.categories[int(cls_id)]
                        # Convert the numpy coordinate array to a Python list of floats.
                        x1, y1, x2, y2 = xyxy.tolist()
                        # Append the detection as a (label, confidence, box) tuple.
                        detections.append((label, float(conf), (x1, y1, x2, y2)))

        # Build a fixed-length feature vector from the list of detections.
        feature_vec = self._build_feature_vec(detections)
        # Return the feature vector and the raw detection list.
        return feature_vec, detections

    def _build_feature_vec(self, detections: list) -> torch.Tensor:
        # Create a zero-initialized feature vector with one element per road category.
        feature_vec = torch.zeros(len(self.categories))
        # Dictionary to accumulate the maximum score for each matched category index.
        category_scores = {}
        # Iterate over each detection to map label strings to category indices.
        for label, score, _ in detections:
            # Check every category against the detection label (case-insensitive substring match).
            for i, cat in enumerate(self.categories):
                # If the category is a substring of the lowercase label or the label is a substring.
                if cat in label.lower() or label.lower() in cat:
                    # Keep the highest confidence score for this category.
                    category_scores[i] = max(category_scores.get(i, 0), score)
        # Fill in the feature vector with the maximum scores per category.
        for idx, score in category_scores.items():
            # Assign the accumulated score to the corresponding position in the feature vector.
            feature_vec[idx] = score
        # Return the populated feature vector.
        return feature_vec


def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    # Create a copy of the input image so the original is not modified.
    img = image.copy()
    # Obtain an ImageDraw object for drawing rectangles and text on the copied image.
    draw = ImageDraw.Draw(img)
    # Attempt to load the DejaVu Sans TrueType font at size 14; fall back to the default bitmap font on failure.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Iterate over each detection, using enumerate to cycle through the color palette.
    for i, (label, score, box) in enumerate(detections):
        # Unpack the four bounding box coordinates.
        x1, y1, x2, y2 = box
        # Select a color from the predefined palette, cycling when there are more boxes than colors.
        color = BOX_COLORS[i % len(BOX_COLORS)]
        # Draw a rectangle outline around the detected object with the chosen color and 2-pixel width.
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # Format the label string to include the detection score with 2 decimal places.
        text = f"{label} ({score:.2f})"
        # Compute the bounding box of the text so we can draw a filled background rectangle behind it.
        bbox = draw.textbbox((x1, y1 - 16), text, font=font)
        # Draw a filled rectangle behind the text using the same color as the detection box.
        draw.rectangle(bbox, fill=color)
        # Draw the label text on top of the filled rectangle in white for readability.
        draw.text((x1, y1 - 16), text, fill="white", font=font)
    # Return the annotated copy of the image.
    return img


def print_features(label, feature_vec, categories, top_k=10):
    # Convert the feature vector to a numpy array for easier iteration and sorting.
    scores = feature_vec.cpu().numpy()
    # Collect (index, score) pairs for all categories that have a non-zero score.
    nonzero = [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0]
    # Sort the non-zero entries by score in descending order.
    nonzero.sort(key=lambda x: x[1], reverse=True)
    # Print a header showing the feature label and how many categories were detected.
    print(f"\n{label} ({len(nonzero)} categories detected):")
    # Print a horizontal separator line for visual clarity.
    print("-" * 60)
    # Print the top_k highest-scoring categories with their scores.
    for idx, score in nonzero[:top_k]:
        print(f"  {categories[idx]:<30s}  score={score:.4f}")
    # If there are more non-zero categories than top_k, indicate how many were omitted.
    if len(nonzero) > top_k:
        print(f"  ... and {len(nonzero) - top_k} more")


def main():
    # Create an ArgumentParser for the feature detection test script with a description.
    parser = argparse.ArgumentParser(
        description="Run feature detection on a single photo, save augmented images and draw bounding boxes."
    )
    # Add a required positional argument for the input image file path.
    parser.add_argument("image", type=str, help="Path to the input photo")
    # Add an optional argument to select the road detection model backbone.
    parser.add_argument(
        "--road-model",
        choices=["grounding_dino", "yolo_world"],
        default="grounding_dino",
    )
    # Add an optional argument to select the vegetation detection model backbone.
    parser.add_argument(
        "--veg-model",
        choices=["clip", "ram++"],
        default="clip",
    )
    # Add an optional argument for the compute device, defaulting to CUDA if available otherwise CPU.
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Add an optional argument (-n) to specify how many augmented variants to generate.
    parser.add_argument(
        "-n",
        type=int,
        default=AUGMENTATIONS_PER_IMAGE,
        help="Number of augmented variants to test (default: AUGMENTATIONS_PER_IMAGE)",
    )
    # Add an optional argument to control how many top-scoring categories are displayed per feature group.
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top-scoring categories to display",
    )
    # Add an optional argument for the output directory where annotated images are saved.
    parser.add_argument(
        "--outdir",
        type=str,
        default="output/feature_test",
        help="Output directory for images",
    )
    # Add an optional argument to control GroundingDINO's box confidence threshold.
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.3,
        help="GroundingDINO box threshold",
    )
    # Add an optional argument to control GroundingDINO's text alignment threshold.
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="GroundingDINO text threshold",
    )
    # Add an optional argument to control YOLOWorld's confidence threshold.
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="YOLOWorld confidence threshold",
    )
    # Parse the command-line arguments into the args namespace.
    args = parser.parse_args()

    # Convert the image path string to a Path object for robust filesystem operations.
    img_path = Path(args.image)
    # Exit early with an error message if the specified image file does not exist.
    if not img_path.exists():
        print(f"Error: image not found: {args.image}")
        return

    # Convert the output directory string to a Path object.
    outdir = Path(args.outdir)
    # Create the output directory (and any missing parent directories) if it does not already exist.
    outdir.mkdir(parents=True, exist_ok=True)
    # Extract the filename stem (without extension) for use in naming output files.
    stem = img_path.stem

    # Import the two vegetation extractor classes: CLIP-based and RAM++ based.
    from features import CLIPBasedVegetationExtractor, RAMPlusExtractor
    # Instantiate the CLIP-based vegetation extractor if the user selected "clip".
    if args.veg_model == "clip":
        veg_extractor = CLIPBasedVegetationExtractor(device=args.device)
    # Otherwise, instantiate the RAM++ vegetation extractor.
    else:
        veg_extractor = RAMPlusExtractor(device=args.device)

    # Print which road detection model is being loaded.
    print(f"Loading road detector: {args.road_model}")
    # Instantiate the GroundingDINO detector if the user selected "grounding_dino".
    if args.road_model == "grounding_dino":
        road_detector = GroundingDINODetector(
            device=args.device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
    # Otherwise, instantiate the YOLO-World detector with the given confidence threshold.
    else:
        road_detector = YOLOWorldDetector(
            device=args.device,
            conf_threshold=args.conf_threshold,
        )
    # Print the dimensionality of the road feature vector.
    print(f"  road feature dim: {len(ROAD_CATEGORIES)}")
    # Print the dimensionality of the vegetation feature vector (obtained from the extractor).
    print(f"  veg feature dim:  {veg_extractor.feature_dim()}")

    # Load the input image from disk and ensure it is in 3-channel RGB format.
    image = Image.open(img_path).convert("RGB")
    # Print the image file path and its pixel dimensions (width x height).
    print(f"\nImage: {img_path}  ({image.size[0]}x{image.size[1]})")

    # Print a header for the original (unaugmented) image feature detection section.
    print("\n" + "=" * 60)
    print("ORIGINAL IMAGE")
    print("=" * 60)

    # Run road feature detection on the original image; returns feature vector and bounding box detections.
    road_feat, road_dets = road_detector.detect(image)
    # Run vegetation feature extraction on the original image.
    veg_feat = veg_extractor.extract(image)

    # Print the top-k road feature scores detected in the original image.
    print_features("Road features", road_feat, ROAD_CATEGORIES, top_k=args.top_k)
    # Print the top-k vegetation feature scores detected in the original image.
    print_features("Vegetation features", veg_feat, VEGETATION_CATEGORIES, top_k=args.top_k)

    # Print the total number of bounding boxes detected in the original image.
    print(f"\nBounding boxes found: {len(road_dets)}")
    # Print details of each detected bounding box: label, confidence score, and coordinates.
    for label, score, box in road_dets:
        print(f"  {label:<30s} score={score:.4f}  box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})")

    # If any road detections were found, draw bounding boxes on the image and save the annotated result.
    if road_dets:
        # Draw colored bounding boxes and labels on a copy of the original image.
        boxed = draw_boxes(image, road_dets)
        # Construct the output file path for the annotated original image.
        boxed_path = outdir / f"{stem}_boxes.png"
        # Save the annotated image to disk.
        boxed.save(boxed_path)
        # Print confirmation that the annotated image was saved.
        print(f"Saved annotated image: {boxed_path}")

    # Initialize the list of all road feature vectors (starting with the original).
    all_road_feats = [road_feat]
    # Initialize the list of all vegetation feature vectors (starting with the original).
    all_veg_feats = [veg_feat]

    # Only generate and process augmented variants if the user requested more than 0.
    if args.n > 0:
        # Print how many augmented variants will be generated.
        print(f"\nGenerating {args.n} augmented variant(s)...")
        # Generate the requested number of augmented image tensors from the original image.
        variants = generate_augmented_variants(image, args.n)

        # Enumerate over the generated variant tensors to process each one.
        for j, variant_tensor in enumerate(variants):
            # Convert the augmented tensor back to a PIL image for saving and display.
            aug_pil = tensor_to_pil(variant_tensor)
            # Construct the output file path for the augmented image.
            aug_path = outdir / f"{stem}_aug{j}.png"
            # Save the augmented PIL image to disk.
            aug_pil.save(aug_path)

            # Run road feature detection on the augmented variant tensor.
            aug_road_feat, aug_road_dets = road_detector.detect(variant_tensor)
            # Run vegetation feature extraction on the augmented variant tensor.
            aug_veg_feat = veg_extractor.extract(variant_tensor)

            # Append the augmented road feature vector to the aggregation list.
            all_road_feats.append(aug_road_feat)
            # Append the augmented vegetation feature vector to the aggregation list.
            all_veg_feats.append(aug_veg_feat)

            # Print a separator header for this augmented variant's results.
            print(f"\n{'=' * 60}")
            print(f"AUGMENTED VARIANT {j + 1}")
            print(f"{'=' * 60}")
            # Print the file path where the augmented image was saved.
            print(f"Saved: {aug_path}")
            # Print how many bounding boxes were detected in this augmented variant.
            print(f"Bounding boxes found: {len(aug_road_dets)}")
            # Print details of each detection: label, score, and bounding box coordinates.
            for label, score, box in aug_road_dets:
                print(f"  {label:<30s} score={score:.4f}  box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})")
            # Print the top-k road feature scores for this augmented variant.
            print_features(
                f"Road features (aug {j + 1})",
                aug_road_feat,
                ROAD_CATEGORIES,
                top_k=args.top_k,
            )
            # Print the top-k vegetation feature scores for this augmented variant.
            print_features(
                f"Vegetation features (aug {j + 1})",
                aug_veg_feat,
                VEGETATION_CATEGORIES,
                top_k=args.top_k,
            )

            # If any road detections were found in the augmentation, draw and save the annotated image.
            if aug_road_dets:
                # Draw bounding boxes and labels on the augmented PIL image.
                aug_boxed = draw_boxes(aug_pil, aug_road_dets)
                # Construct the output path for the annotated augmented image.
                aug_boxed_path = outdir / f"{stem}_aug{j}_boxes.png"
                # Save the annotated augmented image to disk.
                aug_boxed.save(aug_boxed_path)
                # Print confirmation that the annotated image was saved.
                print(f"Saved annotated image: {aug_boxed_path}")

        # Compute the element-wise mean of all road feature vectors across original and augmented images.
        mean_road = torch.stack(all_road_feats).mean(dim=0)
        # Compute the element-wise mean of all vegetation feature vectors across original and augmented images.
        mean_veg = torch.stack(all_veg_feats).mean(dim=0)
        # Print a header for the mean feature results section.
        print(f"\n{'=' * 60}")
        print("MEAN OVER ORIGINAL + AUGMENTED")
        print(f"{'=' * 60}")
        # Print the top-k categories of the mean road feature vector.
        print_features("Road features (mean)", mean_road, ROAD_CATEGORIES, top_k=args.top_k)
        # Print the top-k categories of the mean vegetation feature vector.
        print_features("Vegetation features (mean)", mean_veg, VEGETATION_CATEGORIES, top_k=args.top_k)

    # Print a completion message showing the output directory.
    print(f"\nDone. Output in {outdir}/")


# Standard Python entry-point guard: run main() when the script is executed directly.
if __name__ == "__main__":
    main()
