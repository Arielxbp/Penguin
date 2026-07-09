# Import ABC base class and abstractmethod decorator for defining abstract base classes
from abc import ABC, abstractmethod
# Import Optional type hint for optional return/value annotations
from typing import Optional

# Import PyTorch for tensor operations, GPU acceleration, and neural network utilities
import torch
# Import torchvision transforms (aliased as T) for image conversion and normalization
import torchvision.transforms as T
# Import PIL Image class for loading, manipulating, and converting image files
from PIL import Image
# Import HuggingFace transformers model and processor for zero-shot grounded object detection
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
# Import utility function to download files from HuggingFace Hub repositories
from huggingface_hub import hf_hub_download

# Import configuration constants from the project's config module
from config import (
    # HuggingFace model ID for Grounding DINO object detection model
    GROUNDING_DINO_MODEL,
    # Dimensionality of the road/object feature vector output
    OBJ_FEATURE_DIM,
    # HuggingFace model repo ID for the RAM++ (Recognize Anything Plus) tagging model
    RAM_PLUS_MODEL,
    # List of category names representing road-related objects to detect
    ROAD_CATEGORIES,
    # Dimensionality of the vegetation feature vector output
    VEG_FEATURE_DIM,
    # List of category names representing vegetation types to detect
    VEGETATION_CATEGORIES,
    # Model identifier string for the YOLO-World open-vocabulary object detector
    YOLO_WORLD_MODEL,
)


# Abstract base class defining the interface for all vision feature extractors
class FeatureExtractor(ABC):
    # Decorator marking extract() as an abstract method that subclasses must implement
    @abstractmethod
    # Abstract method to extract a feature tensor from a single PIL image
    def extract(self, image: Image.Image) -> torch.Tensor:
        # Placeholder body for the abstract method; subclasses override this
        pass

    # Default batch extraction method that stacks results from individual image extraction
    def extract_batch(self, images: list) -> torch.Tensor:
        # Apply extract() to each image in the list and stack results into a single tensor
        return torch.stack([self.extract(img) for img in images])

    # Decorator marking feature_dim() as an abstract method that subclasses must implement
    @abstractmethod
    # Abstract method to return the dimensionality of the feature vector produced by this extractor
    def feature_dim(self) -> int:
        # Placeholder body for the abstract method; subclasses override this
        pass


# Feature extractor using Grounding DINO for zero-shot object detection on road scenes
class GroundingDINOExtractor(FeatureExtractor):
    # Initialize the extractor with device selection, detection thresholds, and batch size limit
    def __init__(self, device: str = "cuda", box_threshold: float = 0.3, text_threshold: float = 0.25,
                 max_batch_size: int = 2):
        # Set device to the requested GPU if CUDA is available, otherwise fall back to CPU
        self.device = device if torch.cuda.is_available() else "cpu"
        # Store the bounding box confidence threshold for filtering detections
        self.box_threshold = box_threshold
        # Store the text similarity threshold for filtering grounded detections
        self.text_threshold = text_threshold
        # Store the maximum number of images to process in a single model forward pass
        self.max_batch_size = max_batch_size
        # Load the road object category names from the project configuration
        self.categories = ROAD_CATEGORIES
        # Lazy-loaded model reference; set to None initially, loaded on first use
        self._model = None
        # Lazy-loaded processor reference; set to None initially, loaded on first use
        self._processor = None
        # CLIP image normalization mean values reshaped to (3, 1, 1) for channel-wise broadcasting
        self._clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        # CLIP image normalization standard deviation values reshaped to (3, 1, 1) for channel-wise broadcasting
        self._clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    # Lazily load the Grounding DINO model and processor from HuggingFace on first access
    def _load(self):
        # Check if the model has not been loaded yet
        if self._model is None:
            # Download and instantiate the Grounding DINO processor for input preprocessing
            self._processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
            # Download, instantiate the Grounding DINO model, and move it to the configured device
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                GROUNDING_DINO_MODEL
            ).to(self.device)
            # Set the model to evaluation mode (disables dropout, batch norm updates)
            self._model.eval()

    # Convert a normalized CLIP-space tensor back into a PIL Image for visualization or further processing
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        # Move tensor to CPU and reverse CLIP normalization by multiplying by std and adding mean
        tensor = tensor.cpu() * self._clip_std + self._clip_mean
        # Clamp all pixel values to the valid [0, 1] range
        tensor = torch.clamp(tensor, 0, 1)
        # Convert the (C, H, W) tensor to a PIL Image using torchvision ToPILImage transform
        return T.ToPILImage()(tensor)

    # Return the dimensionality of the output feature vector (number of road categories)
    def feature_dim(self) -> int:
        # Return the configured object feature dimension from config
        return OBJ_FEATURE_DIM

    # Build a fixed-length feature vector from detection results by mapping scores to categories
    def _build_feature_vec(self, result) -> torch.Tensor:
        # Initialize a zero tensor with one element per road category
        feature_vec = torch.zeros(len(self.categories))
        # Only process if the detection result is non-empty and contains detections
        if result and len(result) > 0:
            # Extract the list of predicted label strings from the detection result
            labels = result.get("labels", [])
            # Extract the list of confidence scores from the detection result
            scores = result.get("scores", [])
            # Dictionary to accumulate the maximum confidence per category index
            category_scores = {}
            # Iterate over each detected label and its associated confidence score
            for label, score in zip(labels, scores):
                # Convert the label string to lowercase for case-insensitive matching
                label_lower = label.lower()
                # Check each road category to see if it matches the detected label
                for i, cat in enumerate(self.categories):
                    # Match if the category is a substring of the label or vice versa
                    if cat in label_lower or label_lower in cat:
                        # Store the maximum score seen for this category index
                        category_scores[i] = max(category_scores.get(i, 0), score)
            # Assign the accumulated maximum scores to the corresponding positions in the feature vector
            for idx, score in category_scores.items():
                # Set the feature value at the category index to the accumulated score
                feature_vec[idx] = score
        # Return the constructed feature vector (all zeros if no detections matched)
        return feature_vec

    # Decorator to disable gradient computation and reduce memory usage during inference
    @torch.inference_mode()
    # Extract road object features from a single image using Grounding DINO detection
    def extract(self, image) -> torch.Tensor:
        # Ensure the model and processor are loaded before extraction
        self._load()
        # If the input is a tensor (not a PIL Image), convert it to a PIL Image first
        if not isinstance(image, Image.Image):
            # Convert the normalized tensor back to a PIL image using CLIP inverse normalization
            image = self._tensor_to_pil(image)
        # Build a text query string by joining all category prompts with ". " and appending a period
        text_query = ". ".join([f"a {cat}" for cat in self.categories]) + "."
        # Preprocess the image and text query, convert to PyTorch tensors, and move to the configured device
        inputs = self._processor(images=image, text=text_query, return_tensors="pt").to(self.device)
        # Run the model forward pass to get detection outputs
        outputs = self._model(**inputs)
        # Post-process the raw model outputs into structured detection results with bounding boxes and labels
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )
        # Build and return the category-score feature vector from the first (only) result entry
        return self._build_feature_vec(results[0] if results else None)

    # Decorator to disable gradient computation and reduce memory usage during batch inference
    @torch.inference_mode()
    # Extract road object features from a batch of images using Grounding DINO with batching support
    def extract_batch(self, images: list) -> torch.Tensor:
        # Ensure the model and processor are loaded before extraction
        self._load()
        # Build a text query string by joining all category prompts with ". " and appending a period
        text_query = ". ".join([f"a {cat}" for cat in self.categories]) + "."
        # List to collect PIL images after any necessary tensor-to-PIL conversion
        pil_images = []
        # List to collect target (height, width) sizes for each image for post-processing
        target_sizes = []
        # Convert each input image to PIL format and record its size
        for img in images:
            # If the input is a tensor instead of a PIL Image, convert it first
            if not isinstance(img, Image.Image):
                # Convert the normalized tensor back to a PIL image using CLIP inverse normalization
                img = self._tensor_to_pil(img)
            # Append the PIL image to the list
            pil_images.append(img)
            # Record the (height, width) size of this image for post-processing
            target_sizes.append(img.size[::-1])
        # List to accumulate detection results from each mini-batch
        all_results = []
        # Process images in mini-batches to respect the max_batch_size limit
        for start in range(0, len(pil_images), self.max_batch_size):
            # Compute the end index for this mini-batch slice
            end = start + self.max_batch_size
            # Slice the required number of images for this mini-batch
            sub_images = pil_images[start:end]
            # Slice the corresponding target sizes for this mini-batch
            sub_sizes = target_sizes[start:end]
            # Create a list of text queries (one per image) for batched processing
            texts = [text_query] * len(sub_images)
            # Preprocess the mini-batch of images and text queries, convert to tensors, move to device
            inputs = self._processor(images=sub_images, text=texts, return_tensors="pt").to(self.device)
            # Run the model forward pass on the mini-batch
            outputs = self._model(**inputs)
            # Post-process the raw model outputs into structured detection results
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=sub_sizes,
            )
            # Extend the accumulated results list with results from this mini-batch
            all_results.extend(results)
        # Stack feature vectors from all images into a single batch tensor
        return torch.stack([self._build_feature_vec(r) for r in all_results])


# Feature extractor using YOLO-World for open-vocabulary object detection on road scenes
class YOLOWorldExtractor(FeatureExtractor):
    # Initialize the extractor with device selection, confidence threshold, and optional model loading
    def __init__(self, device: str = "cuda", conf_threshold: float = 0.25):
        # Set device to the requested GPU if CUDA is available, otherwise fall back to CPU
        self.device = device if torch.cuda.is_available() else "cpu"
        # Store the confidence threshold for filtering YOLO detections
        self.conf_threshold = conf_threshold
        # Load the road object category names from the project configuration
        self.categories = ROAD_CATEGORIES
        # Lazy-loaded model reference; set to None initially, loaded on first use
        self._model = None
        # CLIP image normalization mean values reshaped to (3, 1, 1) for channel-wise broadcasting
        self._clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        # CLIP image normalization standard deviation values reshaped to (3, 1, 1) for channel-wise broadcasting
        self._clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        # Attempt to import the YOLOWorld class from the ultralytics package
        try:
            # Import YOLOWorld class from ultralytics for open-vocabulary detection
            from ultralytics import YOLOWorld
            # Store the YOLOWorld class reference for later model instantiation
            self._model_cls = YOLOWorld
        # If the ultralytics package is not installed, set the model class to None
        except ImportError:
            # Set model class to None so extract() can return an empty feature vector gracefully
            self._model_cls = None

    # Lazily load the YOLO-World model with custom classes on first access
    def _load(self):
        # Only load the model if it hasn't been loaded yet and the model class is available
        if self._model is None and self._model_cls is not None:
            # Instantiate the YOLOWorld model with the specified model weights identifier
            self._model = self._model_cls(YOLO_WORLD_MODEL)
            # Move the model to the configured device (GPU or CPU)
            self._model.to(self.device)
            # Set the custom road object categories as the model's detection classes
            self._model.set_classes(self.categories)
            # Set the model to evaluation mode (disables training-specific behaviors)
            self._model.eval()

    # Convert a normalized CLIP-space tensor back into a PIL Image for YOLO inference
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        # Move tensor to CPU and reverse CLIP normalization by multiplying by std and adding mean
        tensor = tensor.cpu() * self._clip_std + self._clip_mean
        # Clamp all pixel values to the valid [0, 1] range
        tensor = torch.clamp(tensor, 0, 1)
        # Convert the (C, H, W) tensor to a PIL Image using torchvision ToPILImage transform
        return T.ToPILImage()(tensor)

    # Return the dimensionality of the output feature vector (number of road categories)
    def feature_dim(self) -> int:
        # Return the configured object feature dimension from config
        return OBJ_FEATURE_DIM

    # Decorator to disable gradient computation and reduce memory usage during inference
    @torch.inference_mode()
    # Extract road object features from a single image using YOLO-World detection
    def extract(self, image) -> torch.Tensor:
        # If the YOLOWorld class failed to import, return a zero feature vector gracefully
        if self._model_cls is None:
            # Return a zero tensor with the same dimensionality as the feature space
            return torch.zeros(self.feature_dim())
        # Ensure the model is loaded before running inference
        self._load()
        # If the input is a tensor instead of a PIL Image, convert it to PIL format
        if not isinstance(image, Image.Image):
            # Convert the normalized tensor back to a PIL image using CLIP inverse normalization
            image = self._tensor_to_pil(image)
        # Run YOLO-World prediction on the image with the configured confidence threshold
        results = self._model.predict(image, conf=self.conf_threshold, verbose=False)
        # Initialize a zero tensor with one element per road category
        feature_vec = torch.zeros(len(self.categories))
        # Only process if prediction returned non-empty results
        if results and len(results) > 0:
            # Access the bounding box predictions from the first (only) result
            boxes = results[0].boxes
            # Only process if there are actual detected boxes
            if boxes is not None and len(boxes) > 0:
                # Extract class ID integers from the detection boxes and convert to numpy
                class_ids = boxes.cls.cpu().int().numpy()
                # Extract confidence scores from the detection boxes and convert to numpy
                confidences = boxes.conf.cpu().numpy()
                # Dictionary to accumulate the maximum confidence per category index
                category_scores = {}
                # Iterate over each detected class ID and its confidence score
                for cls_id, conf in zip(class_ids, confidences):
                    # Convert the class ID to an integer index
                    idx = int(cls_id)
                    # Only process if the index falls within the valid category range
                    if 0 <= idx < len(self.categories):
                        # Store the maximum confidence value seen for this category index
                        category_scores[idx] = max(category_scores.get(idx, 0), float(conf))
                # Assign the accumulated maximum scores to the corresponding positions in the feature vector
                for idx, score in category_scores.items():
                    # Set the feature value at the category index to the accumulated score
                    feature_vec[idx] = score
        # Return the constructed feature vector (all zeros if no detections matched)
        return feature_vec


# Feature extractor using RAM++ (Recognize Anything Plus) for vegetation tagging
class RAMPlusExtractor(FeatureExtractor):
    # Initialize the extractor with device selection and vegetation category list
    def __init__(self, device: str = "cuda"):
        # Set device to the requested GPU if CUDA is available, otherwise fall back to CPU
        self.device = device if torch.cuda.is_available() else "cpu"
        # Load the vegetation category names from the project configuration
        self.veg_categories = VEGETATION_CATEGORIES
        # Lazy-loaded RAM++ model reference; set to None initially, loaded on first use
        self._model = None
        # Lazy-loaded image transformation pipeline reference; set to None initially
        self._transform = None

    # Lazily load the RAM++ model, tag embeddings, and image transform from HuggingFace Hub
    def _load(self):
        # Skip loading if the model has already been loaded
        if self._model is not None:
            return
        # Import the RAM++ model loading function from the recognize_anything package
        from recognize_anything.inference import load_ram_plus

        # Download the RAM++ Swin Transformer model weights file from HuggingFace Hub
        model_path = hf_hub_download(
            repo_id=RAM_PLUS_MODEL,
            filename="ram_plus_swin_large_14m.pth",
        )
        # Download the RAM++ tag embedding weights file from HuggingFace Hub
        embed_path = hf_hub_download(
            repo_id=RAM_PLUS_MODEL,
            filename="ram_plus_tag_embedding_class_4585_des_51.pth",
        )
        # Load the RAM++ model and image transform pipeline using the downloaded weights
        self._model, self._transform = load_ram_plus(
            model_path,
            embed_path,
            device=self.device,
        )
        # Set the model to evaluation mode (disables training-specific behaviors)
        self._model.eval()

    # Return the dimensionality of the output feature vector (number of vegetation categories)
    def feature_dim(self) -> int:
        # Return the configured vegetation feature dimension from config
        return VEG_FEATURE_DIM

    # Decorator to disable gradient computation and reduce memory usage during inference
    @torch.inference_mode()
    # Extract vegetation features from a single image using RAM++ tagging
    def extract(self, image: Image.Image) -> torch.Tensor:
        # Ensure the model, embeddings, and transform are loaded before inference
        self._load()
        # Run RAM++ inference on the image using the loaded model and transform
        result = self._model.inference(image, self._transform)
        # Extract the list of predicted tag strings from the inference result
        tags = result.get("tags", [])
        # Convert all predicted tags to lowercase for case-insensitive matching
        tag_lower = [t.lower() for t in tags]
        # Initialize a zero tensor with one element per vegetation category
        feature_vec = torch.zeros(len(self.veg_categories))
        # Iterate over each vegetation category to check if any predicted tag matches it
        for i, veg_cat in enumerate(self.veg_categories):
            # Check each lowercase predicted tag against the current vegetation category
            for tag in tag_lower:
                # Match if the vegetation category is a substring of the tag or vice versa
                if veg_cat in tag or tag in veg_cat:
                    # Set the feature value to 1.0 to indicate presence of this vegetation type
                    feature_vec[i] = 1.0
                    # Stop checking further tags for this category once a match is found
                    break
        # Return the binary vegetation feature vector
        return feature_vec


# Feature extractor using CLIP for zero-shot vegetation classification
class CLIPBasedVegetationExtractor(FeatureExtractor):
    # Initialize the extractor with device selection and vegetation category list
    def __init__(self, device: str = "cuda"):
        # Set device to the requested GPU if CUDA is available, otherwise fall back to CPU
        self.device = device if torch.cuda.is_available() else "cpu"
        # Load the vegetation category names from the project configuration
        self.veg_categories = VEGETATION_CATEGORIES
        # Lazy-loaded CLIP model reference; set to None initially, loaded on first use
        self._model = None
        # Lazy-loaded CLIP processor reference; set to None initially, loaded on first use
        self._processor = None
        # CLIP image normalization mean values reshaped to (3, 1, 1) for channel-wise broadcasting
        self._clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        # CLIP image normalization standard deviation values reshaped to (3, 1, 1) for channel-wise broadcasting
        self._clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    # Lazily load the CLIP model and processor from HuggingFace on first access
    def _load(self):
        # Skip loading if the model has already been loaded
        if self._model is not None:
            return
        # Import CLIP model and processor classes from HuggingFace transformers
        from transformers import CLIPModel, CLIPProcessor

        # Download and instantiate the CLIP processor from the openai/clip-vit-base-patch32 checkpoint
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Download, instantiate the CLIP model, and move it to the configured device
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        # Set the model to evaluation mode (disables training-specific behaviors)
        self._model.eval()

    # Convert a normalized CLIP-space tensor back into a PIL Image
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        # Move tensor to CPU and reverse CLIP normalization by multiplying by std and adding mean
        tensor = tensor.cpu() * self._clip_std + self._clip_mean
        # Clamp all pixel values to the valid [0, 1] range
        tensor = torch.clamp(tensor, 0, 1)
        # Convert the (C, H, W) tensor to a PIL Image using torchvision ToPILImage transform
        return T.ToPILImage()(tensor)

    # Return the dimensionality of the output feature vector (number of vegetation categories)
    def feature_dim(self) -> int:
        # Return the configured vegetation feature dimension from config
        return VEG_FEATURE_DIM

    # Decorator to disable gradient computation and reduce memory usage during inference
    @torch.inference_mode()
    # Extract vegetation features from a single image using CLIP zero-shot classification
    def extract(self, image) -> torch.Tensor:
        # Ensure the CLIP model and processor are loaded before inference
        self._load()
        # If the input is a tensor instead of a PIL Image, convert it to PIL format
        if not isinstance(image, Image.Image):
            # Convert the normalized tensor back to a PIL image using CLIP inverse normalization
            image = self._tensor_to_pil(image)
        # Build text prompts for CLIP: one "a photo of {category}" string per vegetation category
        texts = [f"a photo of {cat}" for cat in self.veg_categories]
        # Preprocess the text prompts and image, convert to tensors, pad sequences, and move to device
        inputs = self._processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
        # Run the CLIP model forward pass to get image-text similarity logits
        outputs = self._model(**inputs)
        # Extract the logits-per-image tensor for the first (only) image in the batch
        logits_per_image = outputs.logits_per_image[0]
        # Apply softmax to convert logits into probability distribution over vegetation categories
        probs = logits_per_image.softmax(dim=0)
        # Move the probability tensor to CPU and return it
        return probs.cpu()

    # Decorator to disable gradient computation and reduce memory usage during batch inference
    @torch.inference_mode()
    # Extract vegetation features from a batch of images using CLIP zero-shot classification
    def extract_batch(self, images: list) -> torch.Tensor:
        # Ensure the CLIP model and processor are loaded before batch inference
        self._load()
        # List to collect PIL images after any necessary tensor-to-PIL conversion
        pil_images = []
        # Convert each input image to PIL format
        for img in images:
            # If the input is a tensor instead of a PIL Image, convert it first
            if not isinstance(img, Image.Image):
                # Convert the normalized tensor back to a PIL image using CLIP inverse normalization
                img = self._tensor_to_pil(img)
            # Append the PIL image to the list for batched processing
            pil_images.append(img)
        # Build text prompts for CLIP: one "a photo of {category}" string per vegetation category
        texts = [f"a photo of {cat}" for cat in self.veg_categories]
        # Preprocess the text prompts and batch of images, convert to tensors, pad, and move to device
        inputs = self._processor(text=texts, images=pil_images, return_tensors="pt", padding=True).to(self.device)
        # Run the CLIP model forward pass on the batch to get image-text similarity logits
        outputs = self._model(**inputs)
        # Apply softmax along the logits dimension to get probability distributions per image
        probs = outputs.logits_per_image.softmax(dim=-1)
        # Move the probability tensor to CPU and return it
        return probs.cpu()


# Composite extractor that combines a road object extractor and a vegetation extractor
class CompositeFeatureExtractor:
    # Initialize with separate extractor instances for road features and vegetation features
    def __init__(
        self,
        road_extractor: FeatureExtractor,
        veg_extractor: FeatureExtractor,
        device: str = "cuda",
    ):
        # Store the road object feature extractor instance
        self.road_extractor = road_extractor
        # Store the vegetation feature extractor instance
        self.veg_extractor = veg_extractor
        # Set device to the requested GPU if CUDA is available, otherwise fall back to CPU
        self.device = device if torch.cuda.is_available() else "cpu"

    # Decorator to disable gradient computation during feature extraction
    @torch.inference_mode()
    # Extract both road and vegetation features from a single image and return them as a dictionary
    def extract(self, image: Image.Image):
        # Extract road object features using the configured road extractor
        road_features = self.road_extractor.extract(image)
        # Extract vegetation features using the configured vegetation extractor
        veg_features = self.veg_extractor.extract(image)
        # Return a dictionary containing both feature tensors under descriptive keys
        return {
            "road_features": road_features,
            "veg_features": veg_features,
        }

    # Decorator to disable gradient computation during batch feature extraction
    @torch.inference_mode()
    # Extract both road and vegetation features from a batch of images and return them as a dictionary
    def extract_batch(self, images: list):
        # Extract road object features for the entire image batch
        road_features = self.road_extractor.extract_batch(images)
        # Extract vegetation features for the entire image batch
        veg_features = self.veg_extractor.extract_batch(images)
        # Return a dictionary containing both batched feature tensors under descriptive keys
        return {
            "road_features": road_features,
            "veg_features": veg_features,
        }

    # Return the dimensionality of the road object feature vector
    def road_dim(self) -> int:
        # Delegate to the road extractor's feature_dim() method
        return self.road_extractor.feature_dim()

    # Return the dimensionality of the vegetation feature vector
    def veg_dim(self) -> int:
        # Delegate to the vegetation extractor's feature_dim() method
        return self.veg_extractor.feature_dim()


# Factory function to create and configure the appropriate feature extractor instances
def create_feature_extractors(
    road_model: str = "grounding_dino",
    veg_model: str = "clip",
    device: str = "cuda",
):
    # Override device to CPU if the requested GPU is not available
    device = device if torch.cuda.is_available() else "cpu"
    # Select the road object extractor based on the requested model type string
    if road_model == "grounding_dino":
        # Create a GroundingDINOExtractor instance configured for the given device
        road_extractor = GroundingDINOExtractor(device=device)
    elif road_model == "yolo_world":
        # Create a YOLOWorldExtractor instance configured for the given device
        road_extractor = YOLOWorldExtractor(device=device)
    else:
        # Raise an error if an unrecognized road model type is requested
        raise ValueError(f"Unknown road model: {road_model}")

    # Select the vegetation extractor based on the requested model type string
    if veg_model == "ram++":
        # Create a RAMPlusExtractor instance configured for the given device
        veg_extractor = RAMPlusExtractor(device=device)
    elif veg_model == "clip":
        # Create a CLIPBasedVegetationExtractor instance configured for the given device
        veg_extractor = CLIPBasedVegetationExtractor(device=device)
    else:
        # Raise an error if an unrecognized vegetation model type is requested
        raise ValueError(f"Unknown veg model: {veg_model}")

    # Build and return a CompositeFeatureExtractor wrapping the selected road and vegetation extractors
    return CompositeFeatureExtractor(road_extractor, veg_extractor, device=device)
