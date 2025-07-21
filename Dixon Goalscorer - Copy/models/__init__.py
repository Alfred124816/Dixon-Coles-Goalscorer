"""
Dixon-Coles Soccer Prediction Model package.

This package contains the implementation of the Dixon-Coles model
for predicting soccer match outcomes and utilities for data visualization.
"""

from .dixon_coles import DixonColesModel
from .visualize import (
    MatplotlibCanvas, create_heatmap, create_heatmap_for_canvas,
    format_outcome_probabilities, format_over_under,
    create_player_scoring_heatmap_for_canvas
)

__all__ = [
    'DixonColesModel',
    'MatplotlibCanvas',
    'create_heatmap',
    'create_heatmap_for_canvas',
    'format_outcome_probabilities',
    'format_over_under',
    'create_player_scoring_heatmap_for_canvas'
] 