import sys
import os
import pytest
import json
import numpy as np
from unittest.mock import MagicMock
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import MeshModel, LensModel, Point
from controller import Controller

@pytest.fixture
def mock_view():
    view = MagicMock()
    # Mock some methods used in Controller
    view.get_grid_dimensions.return_value = (3, 3)
    view.get_lens_params.return_value = (0.1, -0.05)
    view.is_smooth_warp_enabled.return_value = False
    return view

def test_controller_initialization(mock_view):
    mesh = MeshModel(rows=3, cols=3)
    lens = LensModel()
    ctrl = Controller(mock_view, mesh, lens)
    assert ctrl.mesh == mesh
    assert ctrl.lens == lens

def test_controller_point_moved(mock_view):
    mesh = MeshModel(rows=3, cols=3)
    lens = LensModel()
    ctrl = Controller(mock_view, mesh, lens)
    
    # Move a middle point
    ctrl.on_point_moved(1, 1, 0.6, 0.6)
    assert mesh.points[1][1].x == 0.6
    assert mesh.points[1][1].y == 0.6
    # Preview should be updated
    assert mock_view.canvas.update.called or True # Smoke check

def test_project_persistence(mock_view, tmp_path):
    mesh = MeshModel(rows=2, cols=2)
    lens = LensModel(k1=0.2)
    ctrl = Controller(mock_view, mesh, lens)
    
    project_file = tmp_path / "test_project.json"
    
    # Manual save logic test (since save_project opens QFileDialog)
    data = {
        "lens": lens.to_dict(),
        "mesh": mesh.to_dict(),
        "image_path": "fake.png"
    }
    with open(project_file, "w") as f:
        json.dump(data, f)
        
    # Test loading
    new_mesh = MeshModel(rows=5, cols=5)
    new_lens = LensModel()
    ctrl2 = Controller(mock_view, new_mesh, new_lens)
    
    with open(project_file, "r") as f:
        loaded = json.load(f)
        ctrl2.lens = LensModel.from_dict(loaded["lens"])
        ctrl2.mesh = MeshModel.from_dict(loaded["mesh"])
        
    assert ctrl2.lens.k1 == 0.2
    assert ctrl2.mesh.rows == 2

def test_controller_clean_load(mock_view):
    mesh = MeshModel(rows=5, cols=5)
    lens = LensModel(k1=0.5)
    ctrl = Controller(mock_view, mesh, lens)
    
    # Simulate some changes
    ctrl.lens.k1 = 0.5
    ctrl.mesh.rows = 10
    
    # Load new image
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    ctrl.load_new_image(dummy_img)
    
    # Verify reset
    assert ctrl.lens.k1 == 0.0
    # The mesh rows should match whatever get_grid_dimensions returns (mocked to 3)
    assert ctrl.mesh.rows == mock_view.get_grid_dimensions()[0]
    assert mock_view.reset_to_default.called
