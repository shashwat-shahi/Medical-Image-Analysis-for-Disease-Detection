#!/bin/bash

# Quick start script for medical image analysis
# This script demonstrates how to train and evaluate models

echo "========================================"
echo "Medical Image Analysis - Quick Start"
echo "========================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{skin_cancer,brain_tumor}
mkdir -p checkpoints/{skin_cancer,brain_tumor}
mkdir -p results/{skin_cancer,brain_tumor}

echo ""
echo "Setup completed!"
echo ""
echo "Next steps:"
echo "1. Download datasets from:"
echo "   - Skin Cancer: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"
echo "   - Brain Tumor: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
echo ""
echo "2. Extract datasets to:"
echo "   - data/skin_cancer/"
echo "   - data/brain_tumor/"
echo ""
echo "3. Train models:"
echo "   python train.py --config configs/skin_cancer_config.yaml"
echo "   python train.py --config configs/brain_tumor_config.yaml"
echo ""
echo "4. Run inference:"
echo "   python inference.py --model-path checkpoints/best_model.pth \\"
echo "                      --config-path configs/skin_cancer_config.yaml \\"
echo "                      --image-path path/to/image.jpg \\"
echo "                      --visualize"
echo ""
echo "5. Explore the Jupyter notebook:"
echo "   jupyter notebook notebooks/medical_image_analysis_demo.ipynb"