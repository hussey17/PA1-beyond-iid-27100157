"""Model components for Task 2."""

from task2.models.backbone import ResNet18Backbone, freeze_batchnorm_running_statistics
from task2.models.classifier_head import ClassifierHead
from task2.models.domain_discriminator import DomainDiscriminator

__all__ = [
    "ClassifierHead",
    "DomainDiscriminator",
    "ResNet18Backbone",
    "freeze_batchnorm_running_statistics",
]
