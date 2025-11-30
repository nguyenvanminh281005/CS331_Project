"""
Models initialization
"""
from .arcface_model import ArcFaceModel, create_arcface_model
from .magface_model import MagFaceModel, create_magface_model
from .temporal_model import TemporalAwareModel, create_temporal_model
from .temporal_loss import (
    TemporalContrastiveLoss,
    CombinedLoss,
    TemporalAwareLoss,
    TripletTemporalLoss,
    MagnitudeRegularizationLoss
)

__all__ = [
    'ArcFaceModel',
    'create_arcface_model',
    'MagFaceModel',
    'create_magface_model',
    'TemporalAwareModel',
    'create_temporal_model',
    'TemporalContrastiveLoss',
    'CombinedLoss',
    'TemporalAwareLoss',
    'TripletTemporalLoss',
    'MagnitudeRegularizationLoss',
]
