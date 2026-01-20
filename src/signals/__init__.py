"""Signal generation and trend detection modules"""

from .trend_detector import TrendDetector
from .momentum import MomentumSignals
from .breakout import BreakoutDetector

__all__ = ["TrendDetector", "MomentumSignals", "BreakoutDetector"]
