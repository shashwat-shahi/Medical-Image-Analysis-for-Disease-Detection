#!/usr/bin/env python3
"""
Inference script for medical image classification.
"""

import argparse
import os
import sys
import torch
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.architectures import create_model
from src.data.data_loader import get_transforms
from src.utils.visualization import get_device


def load_and_preprocess_image(image_path: str, image_size: tuple = (224, 224)):
    """Load and preprocess a single image for inference."""
    
    # Load image
    if image_path.lower().endswith(('.nii', '.nii.gz')):
        # Handle NIfTI files (simplified)
        import nibabel as nib
        nii_img = nib.load(image_path)
        data = nii_img.get_fdata()
        
        # For 3D images, take the middle slice
        if len(data.shape) == 3:
            middle_slice = data.shape[2] // 2
            image_data = data[:, :, middle_slice]
        else:
            image_data = data
        
        # Normalize to 0-255 range
        image_data = ((image_data - image_data.min()) / 
                     (image_data.max() - image_data.min()) * 255).astype(np.uint8)
        
        # Convert to RGB
        image_data = np.stack([image_data] * 3, axis=-1)
        image = Image.fromarray(image_data)
    else:
        # Standard image formats
        image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    transform = get_transforms(image_size, augment=False)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image


def predict_single_image(model, image_tensor, device, class_names):
    """Make prediction on a single image."""
    
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
    
    return predicted_class, confidence, probabilities


def main():
    parser = argparse.ArgumentParser(description='Inference for medical image classification')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config-path', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of prediction')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = get_device()
    
    print("=" * 60)
    print("MEDICAL IMAGE ANALYSIS - INFERENCE")
    print("=" * 60)
    print(f"Model: {config['model']['name']}")
    print(f"Image: {args.image_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    try:
        # Load model
        print("Loading model...")
        model = create_model(
            model_name=config['model']['name'],
            num_classes=config['model']['num_classes'],
            pretrained=False,  # We're loading trained weights
            dropout_rate=config['model']['dropout_rate']
        )
        
        # Load trained weights
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"Model loaded from: {args.model_path}")
        
        # Load and preprocess image
        print("Processing image...")
        image_tensor, original_image = load_and_preprocess_image(
            args.image_path, 
            tuple(config['data']['image_size'])
        )
        
        # Make prediction
        print("Making prediction...")
        predicted_class, confidence, probabilities = predict_single_image(
            model, image_tensor, device, config['class_names']
        )
        
        # Print results
        print("=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Predicted Class: {config['class_names'][predicted_class]}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print("\nAll Class Probabilities:")
        for i, (class_name, prob) in enumerate(zip(config['class_names'], probabilities)):
            print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        print("=" * 60)
        
        # Save results to file
        results_file = os.path.join(args.output_dir, 'prediction_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Medical Image Analysis - Prediction Results\n")
            f.write(f"{'='*50}\n")
            f.write(f"Image: {args.image_path}\n")
            f.write(f"Model: {config['model']['name']}\n")
            f.write(f"Predicted Class: {config['class_names'][predicted_class]}\n")
            f.write(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)\n\n")
            f.write("All Class Probabilities:\n")
            for i, (class_name, prob) in enumerate(zip(config['class_names'], probabilities)):
                f.write(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)\n")
        
        print(f"Results saved to: {results_file}")
        
        # Create visualization if requested
        if args.visualize:
            print("Creating visualization...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original image
            ax1.imshow(original_image)
            ax1.set_title(f'Input Image\n{os.path.basename(args.image_path)}')
            ax1.axis('off')
            
            # Prediction probabilities
            colors = plt.cm.Set3(np.linspace(0, 1, len(config['class_names'])))
            bars = ax2.barh(config['class_names'], probabilities, color=colors)
            ax2.set_xlabel('Probability')
            ax2.set_title(f'Prediction: {config["class_names"][predicted_class]}\n'
                         f'Confidence: {confidence:.2f}')
            ax2.set_xlim(0, 1)
            
            # Highlight predicted class
            bars[predicted_class].set_color('red')
            bars[predicted_class].set_alpha(0.8)
            
            # Add probability values on bars
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.3f}', va='center')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(args.output_dir, 'prediction_visualization.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {viz_path}")
            
            plt.show()
        
        # Medical interpretation (simplified)
        print("\nMEDICAL INTERPRETATION:")
        print("-" * 30)
        
        if config['data']['dataset_type'] == 'skin_cancer':
            if config['class_names'][predicted_class] == 'mel':
                print("⚠️  MELANOMA detected - Recommend immediate dermatologist consultation")
            elif config['class_names'][predicted_class] in ['bcc', 'akiec']:
                print("⚠️  Potential malignant lesion - Recommend dermatologist evaluation")
            else:
                print("✓ Likely benign lesion - Monitor for changes")
        
        elif config['data']['dataset_type'] == 'brain_tumor':
            if config['class_names'][predicted_class] == 'notumor':
                print("✓ No tumor detected")
            else:
                tumor_type = config['class_names'][predicted_class]
                print(f"⚠️  {tumor_type.upper()} detected - Recommend neurologist consultation")
        
        print("\n⚠️  IMPORTANT: This AI prediction is for screening purposes only.")
        print("   Always consult with qualified medical professionals for diagnosis.")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())