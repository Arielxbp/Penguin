import argparse
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from dataset import PIL_AUG_TRANSFORM, TENSOR_AUG_TRANSFORM

STREETCLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
STREETCLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def generate_augmented_variants(image: Image.Image, n_variants: int):
    variants = []
    for _ in range(n_variants):
        aug_img = PIL_AUG_TRANSFORM(image)
        aug_tensor = T.ToTensor()(aug_img)
        aug_tensor = TENSOR_AUG_TRANSFORM(aug_tensor)
        aug_tensor = T.Normalize(mean=STREETCLIP_MEAN, std=STREETCLIP_STD)(aug_tensor)
        variants.append(aug_tensor)
    return variants


def denormalize(tensor):
    mean = tensor.new_tensor(STREETCLIP_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(STREETCLIP_STD).view(3, 1, 1)
    return tensor * std + mean


def main():
    parser = argparse.ArgumentParser(description="Generate N augmented copies of an input image.")
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument("-n", type=int, default=3, help="Number of augmented copies to generate")
    parser.add_argument("-o", "--outdir", type=str, default="output/augmented_test", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    stem = Path(args.image).stem

    print(f"Generating {args.n} augmented variants of '{args.image}'...")
    variants = generate_augmented_variants(image, args.n)

    for i, tensor in enumerate(tqdm(variants)):
        denorm = denormalize(tensor)
        denorm = denorm.clamp(0, 1)
        pil_img = T.ToPILImage()(denorm)
        pil_img.save(outdir / f"{stem}_aug{i}.png")

    print(f"Saved {args.n} augmented images to {outdir}/")


if __name__ == "__main__":
    main()
