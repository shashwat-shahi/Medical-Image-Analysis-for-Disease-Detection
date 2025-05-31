#!/usr/bin/env python3
"""
Main training script for medical image classification.
"""

import argparse
import os
import sys
import torch
import yaml
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.architectures import create_model
from src.data.data_loader import create_data_loaders
from src.training.trainer import MedicalImageTrainer
from src.utils.visualization import (
    set_random_seeds, get_device, count_parameters, 
    save_config, save_results, create_medical_report
)


def main():
    parser = argparse.ArgumentParser(description='Train medical image classification model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, 
                       help='Override data directory from config')
    parser.add_argument('--epochs', type=int,
                       help='Override number of epochs from config')
    parser.add_argument('--batch-size', type=int,
                       help='Override batch size from config')
    parser.add_argument('--learning-rate', type=float,
                       help='Override learning rate from config')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Setup device
    device = get_device()
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config['paths']['save_dir'], timestamp)
    results_dir = os.path.join(config['paths']['results_dir'], timestamp)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration
    config_save_path = os.path.join(save_dir, 'config.yaml')
    save_config(config, config_save_path)
    
    print("=" * 60)
    print("MEDICAL IMAGE ANALYSIS - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Dataset: {config['data']['dataset_type']}")
    print(f"Model: {config['model']['name']}")
    print(f"Device: {device}")
    print(f"Save Directory: {save_dir}")
    print(f"Results Directory: {results_dir}")
    print("=" * 60)
    
    try:
        # Create data loaders
        print("Loading data...")
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_type=config['data']['dataset_type'],
            data_dir=config['data']['data_dir'],
            batch_size=config['training']['batch_size'],
            image_size=tuple(config['data']['image_size'])
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Create model
        print("Creating model...")
        model = create_model(
            model_name=config['model']['name'],
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout_rate=config['model']['dropout_rate']
        )
        
        # Get model info
        model_info = count_parameters(model)
        print(f"Total parameters: {model_info['total_parameters']:,}")
        print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
        
        # Create trainer
        trainer = MedicalImageTrainer(
            model=model,
            device=device,
            num_classes=config['model']['num_classes'],
            class_names=config['class_names']
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Train model
        print("Starting training...")
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            save_dir=save_dir,
            early_stopping_patience=config['training']['early_stopping_patience'],
            use_scheduler=config['training']['use_scheduler']
        )
        
        # Evaluate on test set
        print("Evaluating on test set...")
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        test_results = trainer.evaluate(test_loader, best_model_path)
        
        # Save results
        all_results = {
            'training_results': training_results,
            'test_results': test_results,
            'model_info': model_info,
            'config': config
        }
        
        results_path = os.path.join(results_dir, 'results.json')
        save_results(all_results, results_path)
        
        # Create plots
        print("Creating visualizations...")
        
        # Training history plot
        history_plot_path = os.path.join(results_dir, 'training_history.png')
        trainer.plot_training_history(history_plot_path)
        
        # Confusion matrix plot
        cm_plot_path = os.path.join(results_dir, 'confusion_matrix.png')
        trainer.plot_confusion_matrix(test_results, cm_plot_path)
        
        # Create medical report
        print("Generating medical report...")
        report_path = os.path.join(results_dir, 'medical_report.md')
        report = create_medical_report(
            results=test_results,
            model_info=model_info,
            config=config,
            class_names=config['class_names'],
            save_path=report_path
        )
        
        # Print summary
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Best Validation Accuracy: {training_results['best_val_acc']:.2f}%")
        print(f"Test Accuracy: {test_results['accuracy']*100:.2f}%")
        print(f"Test Precision: {test_results['precision']:.4f}")
        print(f"Test Recall: {test_results['recall']:.4f}")
        print(f"Test F1-Score: {test_results['f1_score']:.4f}")
        if test_results['roc_auc']:
            print(f"Test ROC-AUC: {test_results['roc_auc']:.4f}")
        print(f"Training Time: {training_results['training_time']/60:.2f} minutes")
        print(f"Results saved to: {results_dir}")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: Data directory not found - {e}")
        print("Please download the dataset and update the data directory in the config file.")
        print("\nDataset URLs:")
        print("- Skin Cancer MNIST: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("- Brain Tumor MRI: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        return 1
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())