import sys
import os
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import MeshModel, LensModel
from controller import Controller
from pytorch_engine import PyTorchWarpEngine

@pytest.fixture
def mock_view():
    view = MagicMock()
    view.get_grid_dimensions.return_value = (5, 5)
    view.get_lens_params.return_value = (0.0, 0.0)
    view.is_smooth_warp_enabled.return_value = False
    return view

def test_extract_grid_from_filename():
    mesh = MeshModel(rows=5, cols=5)
    lens = LensModel()
    ctrl = Controller(MagicMock(), mesh, lens)
    
    assert ctrl.extract_grid_from_filename("calibration_8x10.png") == (8, 10)
    assert ctrl.extract_grid_from_filename("board_12-15_v2.jpg") == (12, 15)
    assert ctrl.extract_grid_from_filename("checker(7,9).bmp") == (7, 9)
    assert ctrl.extract_grid_from_filename("no_numbers.png") is None
    assert ctrl.extract_grid_from_filename("one_number_10.png") is None

def test_ai_grid_estimation_synthetic():
    engine = PyTorchWarpEngine()
    
    # Create a synthetic grid image (8 rows, 10 cols)
    # White background with black grid lines
    img = np.ones((400, 500, 3), dtype=np.uint8) * 255
    
    rows, cols = 8, 10
    for r in range(rows):
        y = int(r * (400 - 1) / (rows - 1))
        cv2.line(img, (0, y), (500, y), (0, 0, 0), 2)
    for c in range(cols):
        x = int(c * (500 - 1) / (cols - 1))
        cv2.line(img, (x, 0), (x, 400), (0, 0, 0), 2)
        
    est_rows, est_cols = engine.estimate_grid_dimensions(img)
    
    # Check if the estimation is accurate
    assert est_rows == rows
    assert est_cols == cols

def test_intelligent_initialization_flow(mock_view):
    mesh = MeshModel(rows=3, cols=3)
    lens = LensModel()
    ctrl = Controller(mock_view, mesh, lens)
    
    # Simulate loading an image with a specific filename
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Case 1: Filename has priority
    ctrl.load_new_image(dummy_img, path="/data/grid_7x9.png")
    assert ctrl.mesh.rows == 7
    assert ctrl.mesh.cols == 9
    mock_view.set_grid_dimensions.assert_called_with(7, 9)
    
    # Case 2: No metadata, fall back to AI (using the synthetic 8x10 from above)
    img_8x10 = np.ones((400, 500, 3), dtype=np.uint8) * 255
    for r in range(8):
        y = int(r * (400 - 1) / (8 - 1))
        cv2.line(img_8x10, (0, y), (500, y), (0, 0, 0), 2)
    for c in range(10):
        x = int(c * (500 - 1) / (10 - 1))
        cv2.line(img_8x10, (x, 0), (x, 400), (0, 0, 0), 2)
        
    ctrl.load_new_image(img_8x10, path="/data/unknown_grid.png")
    assert ctrl.mesh.rows == 8
    assert ctrl.mesh.cols == 10
    mock_view.set_grid_dimensions.assert_any_call(8, 10)
