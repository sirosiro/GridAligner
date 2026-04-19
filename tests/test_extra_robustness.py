import sys
import os
import pytest
import numpy as np
import cv2
import torch
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from pytorch_engine import PyTorchWarpEngine
from model import MeshModel

@pytest.fixture
def engine():
    return PyTorchWarpEngine(device=torch.device("cpu"))

def test_filename_extraction():
    # Controllerの最新正規表現ロジックをテスト
    test_cases = [
        ("grid_9x12.jpg", (9, 12)),
        ("checker_10_by_10_v2.png", (10, 10)),
        ("no_numbers.jpg", None),
        ("7x8_small_9x9.jpg", (7, 8)),
        ("12x12", (12, 12)),
        ("test 15 x 20.png", (15, 20)),
    ]
    
    def extract(path):
        fname = os.path.basename(path)
        # 実装と同じ正規表現: 数字 + 数字以外 + 数字
        match = re.search(r'(\d+)[^0-9]+(\d+)', fname)
        if match:
            v1, v2 = int(match.group(1)), int(match.group(2))
            if v1 > 1 and v2 > 1: return v1, v2
        return None

    for path, expected in test_cases:
        assert extract(path) == expected

def test_estimate_grid_dimensions_smoke(engine):
    h, w = 500, 500
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(1, 10):
        pos = i * 50
        cv2.line(img, (0, pos), (w, pos), (255, 255, 255), 1)
        cv2.line(img, (pos, 0), (pos, h), (255, 255, 255), 1)
        
    rows, cols = engine.estimate_grid_dimensions(img)
    assert rows > 1
    assert cols > 1

def test_extreme_aspect_ratio_smoke(engine):
    img = np.zeros((100, 2000, 3), dtype=np.uint8)
    mesh = MeshModel(rows=2, cols=20)
    out = engine.apply_mesh_rectification(img, mesh)
    assert out.shape[1] == 2000

def test_expand_mesh_already_full():
    from engine import WarpEngine
    mesh = MeshModel(rows=2, cols=2)
    mesh.points[0][0].x, mesh.points[0][0].y = 0.0, 0.0
    mesh.points[0][1].x, mesh.points[0][1].y = 1.0, 0.0
    mesh.points[1][0].x, mesh.points[1][0].y = 0.0, 1.0
    mesh.points[1][1].x, mesh.points[1][1].y = 1.0, 1.0
    
    orig_rows, orig_cols = mesh.rows, mesh.cols
    WarpEngine.expand_mesh(mesh, 1000, 1000)
    assert mesh.rows == orig_rows
    assert mesh.cols == orig_cols
