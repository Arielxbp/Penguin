# Import the argparse module for command-line argument parsing
import argparse
# Import the Path class from pathlib for cross-platform path handling
from pathlib import Path

# Import torchvision transforms for image augmentation and conversion
import torchvision.transforms as T
# Import the PIL Image module for image loading and saving
from PIL import Image
# Import tqdm for progress bar display
from tqdm import tqdm

# Import PIL-based and tensor-based augmentation transforms from the dataset module
from dataset import PIL_AUG_TRANSFORM, TENSOR_AUG_TRANSFORM

# Define the ImageNet/StreetCLIP mean normalization values
STREETCLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
# Define the ImageNet/StreetCLIP standard deviation normalization values
STREETCLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def generate_augmented_variants(image: Image.Image, n_variants: int):
    # Initialize an empty list to hold augmented tensor variants
    variants = []
    # Iterate n_variants times, generating one augmented copy per iteration
    for _ in range(n_variants):
        # Apply the PIL-based augmentation transform to the input image
        aug_img = PIL_AUG_TRANSFORM(image)
        # Convert the augmented PIL image to a PyTorch tensor
        aug_tensor = T.ToTensor()(aug_img)
        # Apply the tensor-based augmentation transform (e.g., color jitter, blur)
        aug_tensor = TENSOR_AUG_TRANSFORM(aug_tensor)
        # Normalize the tensor using the StreetCLIP mean and std
        aug_tensor = T.Normalize(mean=STREETCLIP_MEAN, std=STREETCLIP_STD)(aug_tensor)
        # Append the augmented tensor to the variants list
        variants.append(aug_tensor)
    # Return the list of augmented tensors
    return variants


def denormalize(tensor):
    # Create a mean tensor on the same device with shape (3, 1, 1) for broadcasting
    mean = tensor.new_tensor(STREETCLIP_MEAN).view(3, 1, 1)
    # Create a std tensor on the same device with shape (3, 1, 1) for broadcasting
    std = tensor.new_tensor(STREETCLIP_STD).view(3, 1, 1)
    # Reverse the normalization: scale by std and add mean
    return tensor * std + mean


def main():
    # Create an argument parser with a description of the script
    parser = argparse.ArgumentParser(description="Generate N augmented copies of an input image.")
    # Add a required positional argument for the input image path
    parser.add_argument("image", type=str, help="Path to the input image")
    # Add an optional argument for the number of augmented copies to generate
    parser.add_argument("-n", type=int, default=3, help="Number of augmented copies to generate")
    # Add an optional argument for the output directory path
    parser.add_argument("-o", "--outdir", type=str, default="output/augmented_test", help="Output directory")
    # Parse the command-line arguments
    args = parser.parse_args()

    # Convert the output directory string to a Path object
    outdir = Path(args.outdir)
    # Create the output directory and any necessary parent directories
    outdir.mkdir(parents=True, exist_ok=True)

    # Open the input image and convert it to RGB format
    image = Image.open(args.image).convert("RGB")
    # Extract the filename stem (without extension) from the input path
    stem = Path(args.image).stem

    # Print a status message indicating the number of variants being generated
    print(f"Generating {args.n} augmented variants of '{args.image}'...")
    # Generate the specified number of augmented tensor variants
    variants = generate_augmented_variants(image, args.n)

    # Iterate over the generated variants with a progress bar
    for i, tensor in enumerate(tqdm(variants)):
        # Denormalize the tensor back to the [0, 1] pixel range
        denorm = denormalize(tensor)
        # Clamp the denormalized values to the valid range [0, 1]
        denorm = denorm.clamp(0, 1)
        # Convert the denormalized tensor back to a PIL image
        pil_img = T.ToPILImage()(denorm)
        # Save the PIL image to the output directory with a numbered suffix
        pil_img.save(outdir / f"{stem}_aug{i}.png")

    # Print a completion message with the number of images saved and output path
    print(f"Saved {args.n} augmented images to {outdir}/")


# Check if this script is being run as the main module
if __name__ == "__main__":
    # Execute the main function
    main()
