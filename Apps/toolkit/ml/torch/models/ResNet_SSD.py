import os
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from abc import ABC, abstractmethod
from loguru import logger
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.ops import box_iou
from pydantic import BaseModel, Field

class SSDConfig(BaseModel):
    """Configuration class for SSD model."""
    backbone: str = Field("resnet34", description="Backbone architecture (resnet18, resnet34, resnet50, resnet101, resnet152)")
    num_classes: int = Field(21, description="Number of object classes + 1 background class")
    input_size: int = Field(300, description="Input image size")
    aspect_ratios: List[List[float]] = Field(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        description="Aspect ratios for anchor boxes"
    )
    feature_maps: List[int] = Field(
        [38, 19, 10, 5, 3, 1],
        description="Feature map sizes"
    )
    min_sizes: List[int] = Field(
        [30, 60, 111, 162, 213, 264],
        description="Minimum sizes for anchor boxes"
    )
    max_sizes: List[int] = Field(
        [60, 111, 162, 213, 264, 315],
        description="Maximum sizes for anchor boxes"
    )
    steps: List[int] = Field(
        [8, 16, 32, 64, 100, 300],
        description="Steps for anchor boxes"
    )
    clip: bool = Field(True, description="Whether to clip anchor boxes")
    nms_threshold: float = Field(0.5, description="Non-maximum suppression threshold")
    confidence_threshold: float = Field(0.01, description="Confidence threshold for detections")
    use_fpn: bool = Field(False, description="Whether to use Feature Pyramid Network")
    fpn_out_channels: int = Field(256, description="Number of output channels for FPN")
    num_extra_layers: int = Field(2, description="Number of extra layers after the backbone")

class BaseBackbone(nn.Module, ABC):
    """Abstract base class for backbone networks."""
    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def get_out_channels(self) -> List[int]:
        pass

class ResNetBackbone(BaseBackbone):
    """ResNet backbone for SSD."""
    def __init__(self, backbone: str, use_fpn: bool = False, fpn_out_channels: int = 256):
        super().__init__()
        self.backbone = backbone
        self.use_fpn = use_fpn
        self.fpn_out_channels = fpn_out_channels
        self.out_channels = self._get_out_channels()
        
        resnet = self._get_resnet_model()
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
            ),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        ])
        
        if self.use_fpn:
            self.fpn = FPNModule(self.out_channels, self.fpn_out_channels)

    def _get_resnet_model(self):
        resnet_models = {
            "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
            "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
            "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
            "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
            "resnet152": (resnet152, ResNet152_Weights.DEFAULT),
        }
        if self.backbone not in resnet_models:
            raise ValueError(f"Unsupported ResNet version: {self.backbone}")
        model_func, weights = resnet_models[self.backbone]
        return model_func(weights=weights)

    def _get_out_channels(self):
        if self.backbone in ["resnet18", "resnet34"]:
            return [64, 64, 128, 256, 512]
        else:  # ResNet50, ResNet101, ResNet152
            return [64, 256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for layer in self.feature_extractor:
            x = layer(x)
            features.append(x)
        
        if self.use_fpn:
            features = self.fpn(features)
        
        return features

    def get_out_channels(self) -> List[int]:
        return self.out_channels if not self.use_fpn else [self.fpn_out_channels] * len(self.out_channels)

def create_backbone(backbone_name: str, **kwargs) -> BaseBackbone:
    """Factory function to create backbone networks."""
    if backbone_name.startswith("resnet"):
        return ResNetBackbone(backbone_name, **kwargs)
    # Add more backbones as needed
    raise ValueError(f"Unsupported backbone: {backbone_name}")

class FPNModule(nn.Module):
    """Feature Pyramid Network module."""
    def __init__(self, in_channels_list: List[int], out_channels: int, num_levels: int = 5):
        super().__init__()
        self.num_levels = num_levels
        self.inner_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, 1) for in_channels in in_channels_list[-num_levels:]]
        )
        self.layer_blocks = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(num_levels)]
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        results = []
        last_inner = self.inner_blocks[-1](x[-1])
        results.append(self.layer_blocks[-1](last_inner))

        for idx in range(self.num_levels - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        return results

class AnchorBoxGenerator:
    """Generates anchor boxes for SSD."""
    def __init__(self, config: SSDConfig):
        self.config = config
        self._create_anchor_generator()

    def _create_anchor_generator(self):
        self.anchor_generator = DefaultBoxGenerator(
            sizes=[(self.config.min_sizes[k], self.config.max_sizes[k]) for k in range(len(self.config.min_sizes))],
            aspect_ratios=self.config.aspect_ratios,
        )

    def generate(self, images: torch.Tensor, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.anchor_generator(images, features)

    def decode_single(self, bbox_pred: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        return self.anchor_generator.decode_single(bbox_pred, anchors)

    def optimize_anchor_boxes(self, dataset: torch.utils.data.Dataset) -> None:
        all_boxes = []
        for _, targets in dataset:
            boxes = targets['boxes'].numpy()
            all_boxes.extend(boxes)

        all_boxes = np.array(all_boxes)
        kmeans = KMeans(n_clusters=len(self.config.min_sizes), random_state=42)
        kmeans.fit(all_boxes[:, 2:4])  # Cluster based on width and height

        self.config.min_sizes = kmeans.cluster_centers_.tolist()
        aspect_ratios = [round(width / height, 2) for width, height in kmeans.cluster_centers_]
        self.config.aspect_ratios = [aspect_ratios] * len(self.config.min_sizes)

        self._create_anchor_generator()

class SSDLoss(nn.Module):
    """Loss function for SSD."""
    def __init__(self, num_classes: int, cls_loss: nn.Module, reg_loss: nn.Module, iou_loss: nn.Module, iou_threshold: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.iou_loss = iou_loss
        self.iou_threshold = iou_threshold

    def forward(self, cls_logits: torch.Tensor, bbox_pred: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        gt_classes = torch.cat([t['labels'] for t in targets], dim=0)
        gt_locations = torch.cat([t['boxes'] for t in targets], dim=0)

        # Match predictions to targets
        matched_idxs = self.match_predictions_to_targets(bbox_pred, gt_locations)

        # Filter out unmatched predictions and targets
        cls_logits = cls_logits[matched_idxs]
        bbox_pred = bbox_pred[matched_idxs]
        gt_classes = gt_classes[matched_idxs]
        gt_locations = gt_locations[matched_idxs]

        # Calculate losses
        cls_loss = self.cls_loss(cls_logits.view(-1, self.num_classes), gt_classes)
        reg_loss = self.reg_loss(bbox_pred, gt_locations)
        iou_loss = self.iou_loss(bbox_pred, gt_locations)

        total_loss = cls_loss + reg_loss + iou_loss

        return {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'iou_loss': iou_loss,
            'total_loss': total_loss
        }

    def match_predictions_to_targets(self, bbox_pred: torch.Tensor, gt_locations: torch.Tensor) -> torch.Tensor:
        # Compute IoU between predictions and targets
        iou = box_iou(bbox_pred, gt_locations)

        # Match predictions to targets based on IoU threshold
        max_iou, matched_idxs = iou.max(dim=1)
        matched_idxs[max_iou < self.iou_threshold] = -1  # Unmatched predictions will have index -1

        return matched_idxs

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced datasets."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure predictions and targets are of the correct shape
        assert predictions.shape[:2] == targets.shape[:2], "Predictions and targets must have the same batch size and number of classes"

        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')

        # Calculate p_t
        pt = torch.exp(-ce_loss)

        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class IoULoss(nn.Module):
    """IoU Loss for bounding box regression."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure the predictions and targets are not empty
        if pred.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        # Compute IoU between predictions and targets
        iou = box_iou(pred, target)

        # Compute the mean IoU of the matched boxes
        loss = 1 - iou.diag().mean()

        return loss

class ResNetSSD(nn.Module):
    """ResNet-based SSD model."""
    def __init__(self, config: SSDConfig):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.anchor_generator = AnchorBoxGenerator(config)
        
        self.backbone = create_backbone(config.backbone, use_fpn=config.use_fpn, fpn_out_channels=config.fpn_out_channels)
        
        out_channels = self.backbone.get_out_channels()
        
        # Add extra layers
        self.extra_layers = self._create_extra_layers(out_channels[-1], config.num_extra_layers)
        out_channels.extend([config.fpn_out_channels] * config.num_extra_layers)
        
        num_anchors = self.anchor_generator.anchor_generator.num_anchors_per_location()
        self.head = SSDHead(config.num_classes, out_channels, num_anchors)
        
        self.loss_function = SSDLoss(
            config.num_classes,
            cls_loss=FocalLoss(),
            reg_loss=nn.SmoothL1Loss(reduction='mean'),
            iou_loss=IoULoss()
        )

        # Initialize weights
        self._initialize_weights()

    def _create_extra_layers(self, in_channels: int, num_layers: int) -> nn.ModuleList:
        layers = []
        for i in range(num_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, self.config.fpn_out_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(self.config.fpn_out_channels, self.config.fpn_out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ))
            in_channels = self.config.fpn_out_channels
        return nn.ModuleList(layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        features = self.backbone(images)
        
        # Add extra layer features
        for layer in self.extra_layers:
            features.append(layer(features[-1]))
        
        cls_logits, bbox_pred = self.head(features)
        
        anchors = self.anchor_generator.generate(images, features)
        
        if self.training and targets is not None:
            losses = self.loss_function(cls_logits, bbox_pred, anchors, targets)
            return losses
        else:
            detections = self.post_process(cls_logits, bbox_pred, anchors)
            return detections

    def post_process(self, cls_logits: torch.Tensor, bbox_pred: torch.Tensor, anchors: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        batch_size = cls_logits.size(0)
        device = cls_logits.device
        
        results = []
        for i in range(batch_size):
            scores = F.softmax(cls_logits[i], dim=-1)
            boxes = self.anchor_generator.decode_single(bbox_pred[i], anchors[i])
            
            # Apply NMS for each class
            keep = torch.zeros(scores.shape[0], dtype=torch.bool, device=device)
            for cl in range(1, self.num_classes):
                scores_cl = scores[:, cl]
                keep_cl = torchvision.ops.nms(boxes, scores_cl, self.config.nms_threshold)
                keep[keep_cl] = True
            
            keep = keep.nonzero().squeeze(1)
            boxes, scores = boxes[keep], scores[keep]
            
            # Get max scores and corresponding labels
            max_scores, labels = scores.max(dim=1)
            
            # Filter by confidence threshold
            keep = torch.where(max_scores > self.config.confidence_threshold)[0]
            boxes, scores, labels = boxes[keep], max_scores[keep], labels[keep]
            
            results.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })
        
        return results

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Perform inference on a single image."""
        self.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self(image)

    def save(self, path: str):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device):
        """Load the model from a file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

class CustomDetectionDataset(Dataset):
    """Custom dataset for object detection."""
    def __init__(self, data_dir: str, annotation_file: str, transform: Optional[Callable] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = self.load_annotations(annotation_file)

    def load_annotations(self, annotation_file: str) -> List[Dict[str, Any]]:
        """
        Load annotations from a file.

        Args:
            annotation_file (str): Path to the annotation file.

        Returns:
            List[Dict[str, Any]]: List of annotations.
        """
        # Implement loading annotations from various formats (COCO, PASCAL VOC, etc.)
        # This is a placeholder implementation
        raise NotImplementedError("Annotation loading not implemented")

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Image and target annotations.
        """
        ann = self.annotations[idx]
        img_path = os.path.join(self.data_dir, ann['file_name'])
        image = Image.open(img_path).convert('RGB')

        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.tensor(ann['labels'], dtype=torch.int64)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        target = {'boxes': boxes, 'labels': labels}
        return image, target

    def apply_transform(self, image: Image.Image, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transformations to the image and bounding boxes.

        Args:
            image (Image.Image): Input image.
            boxes (torch.Tensor): Bounding boxes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image and boxes.
        """
        # Implement the transformation logic here
        # Example: Resize, normalize, etc.
        raise NotImplementedError("Transformation logic not implemented")

class BaseDataTransform(ABC, nn.Module):
    """Abstract base class for data transformations."""

    @abstractmethod
    def forward(self, image: Any, target: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Apply the data transformation.

        Args:
            image (Any): The input image.
            target (Dict[str, Any]): The target dictionary containing bounding boxes, labels, etc.

        Returns:
            Tuple[Any, Dict[str, Any]]: The transformed image and target dictionary.
        """
        pass

class MixUp(BaseDataTransform):
    """MixUp data augmentation."""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, image: torch.Tensor, target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(image.size(0))
        mixed_image = lam * image + (1 - lam) * image[rand_index]

        mixed_target = {
            'boxes': torch.cat([target['boxes'], target['boxes'][rand_index]]),
            'labels': torch.cat([target['labels'], target['labels'][rand_index]])
        }

        return mixed_image, mixed_target

class CutMix(BaseDataTransform):
    """CutMix data augmentation."""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, image: torch.Tensor, target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(image.size(0))
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.size(), lam)

        mixed_image = image.clone()
        mixed_image[:, bbx1:bbx2, bby1:bby2] = image[rand_index, bbx1:bbx2, bby1:bby2]

        mixed_target = {
            'boxes': torch.cat([target['boxes'], target['boxes'][rand_index]]),
            'labels': torch.cat([target['labels'], target['labels'][rand_index]])
        }

        return mixed_image, mixed_target

    def rand_bbox(self, size: Tuple[int, int, int], lam: float) -> Tuple[int, int, int, int]:
        W, H = size[1], size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

class Mosaic(BaseDataTransform):
    """Mosaic data augmentation."""
    def __init__(self, size: int = 300):
        super().__init__()
        self.size = size

    def forward(self, image: torch.Tensor, target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Implement Mosaic augmentation
        # This is a placeholder implementation
        return image, target

class AugmentationPipeline:
    """Pipeline for data augmentation."""
    def __init__(self, augmentations: List[BaseDataTransform]):
        self.augmentations = augmentations

    def __call__(self, image: torch.Tensor, target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        for aug in self.augmentations:
            image, target = aug(image, target)
        return image, target

def get_transform(train: bool = True) -> AugmentationPipeline:
    """Get the transformation pipeline for training or evaluation."""
    if train:
        return AugmentationPipeline([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomCrop((300, 300), padding=4),
            transforms.RandomChoice([
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.RandomAutocontrast(),
                transforms.RandomEqualize()
            ]),
            transforms.RandomApply([MixUp(alpha=0.2)], p=0.5),
            transforms.RandomApply([CutMix(alpha=1.0)], p=0.5),
            transforms.RandomApply([Mosaic()], p=0.5),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return AugmentationPipeline([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class MetricsSSD:
    def __init__(self, model: nn.Module, data_loader: DataLoader):
        self.model = model
        self.data_loader = data_loader

    def calculate_map(self, iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]) -> float:
        """Calculate mean Average Precision (mAP) for the model."""
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.data_loader:
                predictions = self.model(images)
                all_predictions.extend(predictions)
                all_targets.extend(targets)

        maps = []
        for iou_threshold in iou_thresholds:
            ap_per_class = self.calculate_ap_per_class(all_predictions, all_targets, iou_threshold)
            maps.append(np.mean(ap_per_class))

        return np.mean(maps)

    def calculate_ap_per_class(self, predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]], iou_threshold: float) -> List[float]:
        """Calculate Average Precision (AP) per class."""
        num_classes = max(t['labels'].max().item() for t in targets) + 1
        ap_per_class = [0.0] * num_classes

        for cls in range(num_classes):
            # Collect all predictions and targets for the current class
            pred_boxes = []
            pred_scores = []
            true_boxes = []

            for pred in predictions:
                cls_mask = pred['labels'] == cls
                pred_boxes.extend(pred['boxes'][cls_mask].tolist())
                pred_scores.extend(pred['scores'][cls_mask].tolist())

            for target in targets:
                cls_mask = target['labels'] == cls
                true_boxes.extend(target['boxes'][cls_mask].tolist())

            # Sort predictions by score in descending order
            pred_boxes = [x for _, x in sorted(zip(pred_scores, pred_boxes), reverse=True)]
            pred_scores.sort(reverse=True)

            # Calculate True Positives (TP) and False Positives (FP)
            tp = [0] * len(pred_boxes)
            fp = [0] * len(pred_boxes)
            matched = set()

            for i, pred_box in enumerate(pred_boxes):
                best_iou = 0
                best_idx = -1
                for j, true_box in enumerate(true_boxes):
                    if j in matched:
                        continue
                    iou = box_iou(torch.tensor([pred_box]), torch.tensor([true_box])).item()
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                if best_iou >= iou_threshold:
                    tp[i] = 1
                    matched.add(best_idx)
                else:
                    fp[i] = 1

            # Calculate Precision and Recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recalls = tp_cumsum / len(true_boxes)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

            # Calculate Average Precision (AP)
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            for i in range(len(precisions) - 1, 0, -1):
                precisions[i - 1] = max(precisions[i - 1], precisions[i])

            ap = 0.0
            for i in range(1, len(recalls)):
                ap += (recalls[i] - recalls[i - 1]) * precisions[i]

            ap_per_class[cls] = ap

        return ap_per_class

def train_one_epoch(
    model: ResNetSSD,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    num_epochs: int
) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    # Learning rate scheduler with warm-up
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(data_loader) - 1)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        if batch_idx < warmup_iters:
            warmup_scheduler.step()
        
        total_loss += losses.item()
        
        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{num_batches}, Loss: {losses.item():.4f}")
    
    scheduler.step()
    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model: ResNetSSD, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Validate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    avg_loss = total_loss / num_batches
    return {"val_loss": avg_loss}

def train(
    model: ResNetSSD,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device
):
    """Train the model for multiple epochs."""
    best_map = 0.0
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, num_epochs)
        val_loss = validate(model, val_loader, device)['val_loss']
        
        map_score = MetricsSSD().calculate_map(model, val_loader)
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, mAP: {map_score:.4f}")
        
        if map_score > best_map:
            best_map = map_score
            model.save('best_model.pth')
            logger.info(f"New best model saved with mAP: {best_map:.4f}")

    logger.info(f"Training completed. Best mAP: {best_map:.4f}")

class InferencePipeline:
    """Pipeline for performing inference with the trained model."""
    def __init__(self, model: ResNetSSD, transform: transforms.Compose, device: torch.device):
        self.model = model.to(device)
        self.transform = transform
        self.device = device

    def __call__(self, image: Image.Image) -> List[Dict[str, torch.Tensor]]:
        self.model.eval()
        with torch.no_grad():
            x = self.transform(image).unsqueeze(0).to(self.device)
            predictions = self.model(x)
        return predictions[0]

# Add any additional utility functions or classes here
