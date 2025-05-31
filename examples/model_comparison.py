"""
Example script demonstrating model comparison across different architectures.
"""

import os
import sys
import torch
import yaml
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.architectures import create_model, MODEL_CONFIGS
from src.utils.visualization import (
    set_random_seeds, get_device, count_parameters,
    create_model_comparison_plot, save_results
)


def compare_models():
    """Compare different model architectures."""
    
    # Set random seeds
    set_random_seeds(42)
    
    # Get device
    device = get_device()
    
    # Configuration
    num_classes = 7  # Skin cancer classes
    
    print("=" * 60)
    print("MODEL ARCHITECTURE COMPARISON")
    print("=" * 60)
    
    comparison_results = {}
    
    # Test different model configurations
    models_to_test = [
        'resnet18', 'resnet50', 'resnet101',
        'densenet121', 'densenet169',
        'efficientnet_b0', 'efficientnet_b3',
        'vit_small_patch16_224', 'vit_base_patch16_224'
    ]
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        try:
            # Create model
            start_time = time.time()
            model = create_model(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=True
            )
            creation_time = time.time() - start_time
            
            # Get model info
            model_info = count_parameters(model)
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                start_time = time.time()
                _ = model(dummy_input)
                inference_time = time.time() - start_time
            
            # Store results
            comparison_results[model_name] = {
                'total_parameters': model_info['total_parameters'],
                'trainable_parameters': model_info['trainable_parameters'],
                'model_size_mb': model_info['total_parameters'] * 4 / (1024**2),
                'creation_time': creation_time,
                'inference_time': inference_time * 1000,  # Convert to ms
                'status': 'success'
            }
            
            print(f"  ✓ Parameters: {model_info['total_parameters']:,}")
            print(f"  ✓ Size: {model_info['total_parameters'] * 4 / (1024**2):.1f} MB")
            print(f"  ✓ Creation time: {creation_time:.3f}s")
            print(f"  ✓ Inference time: {inference_time * 1000:.2f}ms")
            
        except Exception as e:
            comparison_results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"  ✗ Failed: {e}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/model_comparison_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    save_results(comparison_results, os.path.join(results_dir, 'comparison_results.json'))
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    successful_models = {k: v for k, v in comparison_results.items() if v['status'] == 'success'}
    
    if successful_models:
        # Find best models by different criteria
        smallest_model = min(successful_models.items(), key=lambda x: x[1]['total_parameters'])
        fastest_model = min(successful_models.items(), key=lambda x: x[1]['inference_time'])
        
        print(f"Smallest model: {smallest_model[0]} ({smallest_model[1]['total_parameters']:,} params)")
        print(f"Fastest model: {fastest_model[0]} ({fastest_model[1]['inference_time']:.2f}ms)")
        
        print(f"\nResults saved to: {results_dir}")
        
        # Create comparison plot (if matplotlib is available)
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            models = list(successful_models.keys())
            params = [successful_models[m]['total_parameters'] for m in models]
            times = [successful_models[m]['inference_time'] for m in models]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Parameter count comparison
            bars1 = ax1.bar(range(len(models)), params, alpha=0.7)
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Parameters (millions)')
            ax1.set_title('Model Size Comparison')
            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels(models, rotation=45, ha='right')
            
            # Add value labels
            for bar, param in zip(bars1, params):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{param/1e6:.1f}M', ha='center', va='bottom')
            
            # Inference time comparison
            bars2 = ax2.bar(range(len(models)), times, alpha=0.7, color='orange')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Inference Time (ms)')
            ax2.set_title('Inference Speed Comparison')
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels(models, rotation=45, ha='right')
            
            # Add value labels
            for bar, time_val in zip(bars2, times):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{time_val:.1f}ms', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {os.path.join(results_dir, 'model_comparison.png')}")
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    else:
        print("No models were successfully tested!")
    
    return comparison_results


if __name__ == "__main__":
    results = compare_models()