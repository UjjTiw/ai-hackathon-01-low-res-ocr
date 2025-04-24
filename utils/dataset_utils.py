import os
import glob
import random
import unicodedata
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# --- Real-ESRGAN modules from local clone ---
import sys
sys.path.append("Real-ESRGAN")  # Change this to your repo path if different
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

LABELS = [
    "玄関", "バルコニー", "浴室", "トイレ", "収納",
    "洋室", "クローゼット", "廊下", "ホール", "和室",
]
label_to_idx = {
    unicodedata.normalize("NFC", label): i for i, label in enumerate(LABELS)
}
idx_to_label = {i: label for label, i in label_to_idx.items()}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageLabelDataset(Dataset):
    def __init__(self, image_dir, transform=None, with_label=True):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        self.with_label = with_label
        self.device = torch.device("cuda")

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path='Real-ESRGAN/weights/RealESRGAN_x4plus.pth',  # Make sure this exists
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=not self.device.type == 'mps',  # Avoid half precision on MPS
        )
        
        # Pre-process all images with tqdm progress bar
        print("Pre-processing images with Real-ESRGAN upscaling...")
        self.processed_images = {}
        for path in tqdm(self.image_paths, desc="Upscaling images"):
            self._preprocess_image(path)

    # In dataset_utils.py
    def _preprocess_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)
            
            # Add image enhancement before upscaling
            from PIL import ImageEnhance, ImageFilter
            img = Image.fromarray(img_np)
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            # Sharpen edges - helpful for text recognition
            img = img.filter(ImageFilter.SHARPEN)
            img_np = np.array(img)
            
            output, _ = self.upsampler.enhance(img_np, outscale=1)
            self.processed_images[path] = Image.fromarray(output)
        except Exception as e:
            print(f"ERROR: Upscaling failed for {path}: {e}")
            self.processed_images[path] = Image.open(path).convert("RGB")
    def __len__(self):
        return len(self.image_paths)

    # In dataset_utils.py - update ImageLabelDataset.__getitem__
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        
        # Use the pre-processed image
        img = self.processed_images[path]
        
        # Convert to grayscale and enhance text visibility
        img = img.convert("L")
        
        # Text enhancement for character recognition
        img = ImageOps.autocontrast(img, cutoff=2)
        
        img = self.transform(img)

        if self.with_label:
            filename = os.path.basename(path)
            try:
                # Normalize Japanese text properly
                label_name = unicodedata.normalize("NFC", filename.split("_")[0])
                label = label_to_idx[label_name]
            except (IndexError, KeyError):
                raise ValueError(f"ファイル名 '{filename}' からラベルが抽出できませんでした。")
            return img, label
        else:
            return img, os.path.basename(path)