#!/bin/bash

set -e

echo "🚀 Starting Real-ESRGAN setup in conda environment: ocr_env"

# conda activate ocr_env

# Step 1: Clone Real-ESRGAN if not already cloned
if [ ! -d "Real-ESRGAN" ]; then
    echo "📥 Cloning Real-ESRGAN..."
    git clone https://github.com/xinntao/Real-ESRGAN.git
else
    echo "📁 Real-ESRGAN already exists. Skipping clone."
fi

cd Real-ESRGAN

# Step 2: Install required packages
echo "📦 Installing Python dependencies into ocr_env..."
pip install basicsr facexlib gfpgan
pip install torch torchvision numpy pandas scikit-learn tqdm Pillow torchsummary

# Step 3: Install Real-ESRGAN in develop mode
echo "🔧 Setting up Real-ESRGAN (dev mode)..."
python setup.py develop

# Step 4: Go back and run the patch script
cd ..
echo "🩹 Patching torchvision import..."
python patch_import_bug.py

echo "✅ Setup completed successfully!"
