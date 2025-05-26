"""
Deep learning model architectures for medical image analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List
import timm


class ResNetMedical(nn.Module):
    """ResNet architecture for medical image classification."""
    
    def __init__(self, num_classes: int, pretrained: bool = True, 
                 architecture: str = 'resnet50', dropout_rate: float = 0.5):
        super(ResNetMedical, self).__init__()
        
        # Load pretrained ResNet
        if architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif architecture == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feat_dim = 512
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feat_dim = 2048
        elif architecture == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet architecture: {architecture}")
        
        # Replace the original classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class DenseNetMedical(nn.Module):
    """DenseNet architecture for medical image classification."""
    
    def __init__(self, num_classes: int, pretrained: bool = True,
                 architecture: str = 'densenet121', dropout_rate: float = 0.5):
        super(DenseNetMedical, self).__init__()
        
        # Load pretrained DenseNet
        if architecture == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            feat_dim = 1024
        elif architecture == 'densenet169':
            self.backbone = models.densenet169(pretrained=pretrained)
            feat_dim = 1664
        elif architecture == 'densenet201':
            self.backbone = models.densenet201(pretrained=pretrained)
            feat_dim = 1920
        else:
            raise ValueError(f"Unsupported DenseNet architecture: {architecture}")
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        
        output = self.classifier(features)
        return output


class VisionTransformerMedical(nn.Module):
    """Vision Transformer for medical image classification."""
    
    def __init__(self, num_classes: int, pretrained: bool = True,
                 architecture: str = 'vit_base_patch16_224', dropout_rate: float = 0.1):
        super(VisionTransformerMedical, self).__init__()
        
        # Load pretrained Vision Transformer using timm
        self.backbone = timm.create_model(
            architecture, 
            pretrained=pretrained, 
            num_classes=0,  # Remove head
            global_pool='avg'
        )
        
        # Get feature dimension
        feat_dim = self.backbone.num_features
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class EfficientNetMedical(nn.Module):
    """EfficientNet architecture for medical image classification."""
    
    def __init__(self, num_classes: int, pretrained: bool = True,
                 architecture: str = 'efficientnet_b0', dropout_rate: float = 0.3):
        super(EfficientNetMedical, self).__init__()
        
        # Load pretrained EfficientNet using timm
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,  # Remove head
            global_pool='avg'
        )
        
        # Get feature dimension
        feat_dim = self.backbone.num_features
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class ConvNeXtMedical(nn.Module):
    """ConvNeXt architecture for medical image classification."""
    
    def __init__(self, num_classes: int, pretrained: bool = True,
                 architecture: str = 'convnext_base', dropout_rate: float = 0.2):
        super(ConvNeXtMedical, self).__init__()
        
        # Load pretrained ConvNeXt using timm
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,  # Remove head
            global_pool='avg'
        )
        
        # Get feature dimension
        feat_dim = self.backbone.num_features
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def create_model(model_name: str, num_classes: int, pretrained: bool = True, 
                 **kwargs) -> nn.Module:
    """Factory function to create medical image classification models."""
    
    model_name = model_name.lower()
    
    if 'resnet' in model_name:
        return ResNetMedical(
            num_classes=num_classes,
            pretrained=pretrained,
            architecture=model_name,
            **kwargs
        )
    elif 'densenet' in model_name:
        return DenseNetMedical(
            num_classes=num_classes,
            pretrained=pretrained,
            architecture=model_name,
            **kwargs
        )
    elif 'vit' in model_name or 'vision_transformer' in model_name:
        return VisionTransformerMedical(
            num_classes=num_classes,
            pretrained=pretrained,
            architecture=model_name,
            **kwargs
        )
    elif 'efficientnet' in model_name:
        return EfficientNetMedical(
            num_classes=num_classes,
            pretrained=pretrained,
            architecture=model_name,
            **kwargs
        )
    elif 'convnext' in model_name:
        return ConvNeXtMedical(
            num_classes=num_classes,
            pretrained=pretrained,
            architecture=model_name,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_model_info(model: nn.Module) -> dict:
    """Get information about the model."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }


# Model configurations for different use cases
MODEL_CONFIGS = {
    'lightweight': {
        'resnet': 'resnet18',
        'densenet': 'densenet121',
        'efficientnet': 'efficientnet_b0',
        'vit': 'vit_small_patch16_224'
    },
    'balanced': {
        'resnet': 'resnet50',
        'densenet': 'densenet121',
        'efficientnet': 'efficientnet_b3',
        'vit': 'vit_base_patch16_224'
    },
    'high_accuracy': {
        'resnet': 'resnet101',
        'densenet': 'densenet201',
        'efficientnet': 'efficientnet_b7',
        'vit': 'vit_large_patch16_224'
    }
}