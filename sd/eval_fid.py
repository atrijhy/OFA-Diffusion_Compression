# compute_fid_sd.py  —  Compute FID between generated images and COCO val reference
#
# Usage:
#   python compute_fid_sd.py \
#       --gen_dir outputs/ofa_sd_generated \
#       --ref_dir data/coco2014/val2014 \
#       --batch_size 64 --device cuda
#
# If ref_dir contains raw COCO images (varying sizes), they will be
# resized to 299x299 on-the-fly (InceptionV3 input size).

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.functional import adaptive_avg_pool2d

# Use the pytorch-fid InceptionV3 bundled in this directory.
# It downloads FID-specific weights (TF-ported) and handles resize+normalisation
# internally (input: [0, 1] float32 → internal scale to [-1, 1]).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from tools.inception import InceptionV3


def parse_args():
    parser = argparse.ArgumentParser(description='Compute FID for OFA-pruned SD generation')
    parser.add_argument('--gen_dir', type=str, required=True,
                        help='Directory with generated images')
    parser.add_argument('--ref_dir', type=str, required=True,
                        help='Directory with reference images (e.g. COCO val2014)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dims', type=int, default=2048,
                        help='InceptionV3 feature dimensionality')
    parser.add_argument('--max_ref_images', type=int, default=None,
                        help='Max reference images to use (None = all)')
    return parser.parse_args()


class ImageFolderDataset(Dataset):
    """Simple image folder dataset with on-the-fly resize."""

    EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}

    def __init__(self, root, transform=None, max_images=None):
        self.root = Path(root)
        self.transform = transform

        self.paths = sorted([
            p for p in self.root.iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        ])

        if max_images is not None:
            self.paths = self.paths[:max_images]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def get_inception_model(device):
    """Load FID-specific InceptionV3 (pytorch-fid weights, pool3 features).

    Uses the TF-ported FID weights from inception.py — these give FID numbers
    directly comparable with published results and the pytorch-fid CLI tool.
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])   # resize_input=True, normalize_input=True by default
    model.eval().to(device)
    return model


@torch.no_grad()
def compute_features(model, dataloader, device, desc="Computing features"):
    """Extract InceptionV3 pool3 features (2048-d) from a dataloader."""
    all_features = []
    for batch in tqdm(dataloader, desc=desc):
        batch = batch.to(device)

        # inception.py returns a list of feature maps; [0] is the requested block
        out = model(batch)[0]   # [B, 2048, 1, 1] or larger spatial map

        if out.shape[2] != 1 or out.shape[3] != 1:
            out = adaptive_avg_pool2d(out, output_size=(1, 1))

        all_features.append(out.squeeze(3).squeeze(2).cpu().numpy())   # [B, 2048]

    return np.concatenate(all_features, axis=0)


def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute FID between two Gaussians."""
    from scipy import linalg

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        print(f"WARNING: fid calculation produces singular product; adding {eps} to diagonal of cov estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Transform: ToTensor only → [0, 1] range.
    # The FID InceptionV3 (from inception.py) internally resizes to 299×299 and
    # normalises to [-1, 1], so we must NOT apply extra resize / normalisation here.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Datasets
    print(f"Loading generated images from {args.gen_dir}...")
    gen_dataset = ImageFolderDataset(args.gen_dir, transform=transform)
    print(f"  Found {len(gen_dataset)} generated images")

    print(f"Loading reference images from {args.ref_dir}...")
    ref_dataset = ImageFolderDataset(args.ref_dir, transform=transform, max_images=args.max_ref_images)
    print(f"  Found {len(ref_dataset)} reference images")

    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)
    ref_loader = DataLoader(ref_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)

    # Load InceptionV3
    print("Loading InceptionV3...")
    inception = get_inception_model(device)

    # Compute features
    gen_features = compute_features(inception, gen_loader, device, "Generated features")
    ref_features = compute_features(inception, ref_loader, device, "Reference features")

    print(f"Feature shapes: generated={gen_features.shape}, reference={ref_features.shape}")

    # Compute statistics
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    mu_ref = np.mean(ref_features, axis=0)
    sigma_ref = np.cov(ref_features, rowvar=False)

    # Compute FID
    fid = compute_fid(mu_gen, sigma_gen, mu_ref, sigma_ref)

    print(f"\n{'='*40}")
    print(f"  FID: {fid:.2f}")
    print(f"{'='*40}")
    print(f"  Generated: {len(gen_dataset)} images from {args.gen_dir}")
    print(f"  Reference: {len(ref_dataset)} images from {args.ref_dir}")

    # Save result
    result_path = os.path.join(args.gen_dir, 'fid_result.json')
    result = {
        'fid': fid,
        'num_generated': len(gen_dataset),
        'num_reference': len(ref_dataset),
        'gen_dir': args.gen_dir,
        'ref_dir': args.ref_dir,
    }
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {result_path}")


if __name__ == '__main__':
    main()
