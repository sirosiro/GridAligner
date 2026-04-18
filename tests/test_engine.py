import sys
import os
import pytest
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import MeshModel, LensModel, Point
from engine import WarpEngine

def test_sync_perspective():
    # Setup a slightly distorted mesh
    mesh = MeshModel(rows=3, cols=3)
    # Move corners
    mesh.points[0][0].x = 0.1
    mesh.points[0][0].y = 0.1
    mesh.points[0][2].x = 0.9
    mesh.points[0][2].y = 0.1
    
    WarpEngine.sync_perspective(mesh)
    
    # After sync, the middle point (1,1) should be at (0.5, 0.5) because it's a linear homography of a square
    assert pytest.approx(mesh.points[1][1].x, abs=0.01) == 0.5
    assert pytest.approx(mesh.points[1][1].y, abs=0.01) == 0.5

def test_expand_mesh_limit():
    # Setup a small 2x2 mesh in the center
    mesh = MeshModel(rows=2, cols=2)
    # Scale it to 0.4 - 0.6 range
    for r in range(2):
        for c in range(2):
            mesh.points[r][c].x = 0.4 + c * 0.2
            mesh.points[r][c].y = 0.4 + r * 0.2
    
    # Expand it in a 1000x1000 image
    # It should add rows/cols until FOV limit
    WarpEngine.expand_mesh(mesh, 1000, 1000)
    
    # Since original was 0.4-0.6 (20% width), it should expand several times.
    # Total points should cover nearly the whole image.
    assert mesh.rows > 2
    assert mesh.cols > 2
    
    # Check that points are within 0-1 range
    for r in range(mesh.rows):
        for c in range(mesh.cols):
            assert 0 <= mesh.points[r][c].x <= 1.0
            assert 0 <= mesh.points[r][c].y <= 1.0

def test_apply_lens_distortion():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.line(img, (0, 50), (100, 50), (255, 255, 255), 1) # Straight line
    
    lens = LensModel(k1=0.5, k2=0.0) # Barrel distortion
    distorted = WarpEngine.apply_lens_distortion(img, lens)
    
    # The line should now be curved.
    # We just check that the output has the same shape and is not identical to input.
    assert distorted.shape == img.shape
    assert not np.array_equal(img, distorted)

def test_get_mesh_map():
    mesh = MeshModel(rows=2, cols=2)
    # Correct signature: (mesh, src_w, src_h, dst_w, dst_h)
    mx, my = WarpEngine.get_mesh_map(mesh, 100, 100, 100, 100)
    assert mx.shape == (100, 100)
    assert my.shape == (100, 100)
    # Corner check in the map
    assert mx[0, 0] == 0.0
    assert mx[99, 99] == 99.0

def test_engine_rectification_smoke():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mesh = MeshModel(rows=2, cols=2)
    # Simple rectification
    out = WarpEngine.apply_mesh_rectification(img, mesh)
    assert out.shape == (100, 100, 3)

def test_engine_process_preview_smoke():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mesh = MeshModel(rows=2, cols=2)
    lens = LensModel()
    out = WarpEngine.process_preview(img, mesh, lens, rectified=True)
    assert out is not None
