"""
UAV Landing System with Neurosymbolic Memory

A production-ready UAV landing detection system combining computer vision 
with neurosymbolic memory for robust performance in challenging scenarios.
"""

__version__ = "1.0.0"
__author__ = "MegaPanchamZ"

from .detector import UAVLandingDetector
from .memory import NeuroSymbolicMemory
from .types import LandingResult, MemoryZone

__all__ = [
    "UAVLandingDetector",
    "NeuroSymbolicMemory", 
    "LandingResult",
    "MemoryZone"
]
