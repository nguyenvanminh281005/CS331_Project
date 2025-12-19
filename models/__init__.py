"""
Models initialization
"""
from .arcface_model import ArcFaceModel, create_arcface_model
from .magface_model import MagFaceModel, create_magface_model

__all__ = [
    'ArcFaceModel',
    'create_arcface_model',
    'MagFaceModel',
    'create_magface_model',
]
