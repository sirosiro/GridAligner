import sys
import os
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import Point, LensModel, MeshModel

def test_point_creation():
    p = Point(0.5, 0.8)
    assert p.x == 0.5
    assert p.y == 0.8

def test_lens_model_persistence():
    lens = LensModel(k1=0.1, k2=-0.05)
    data = lens.to_dict()
    assert data['k1'] == 0.1
    assert data['k2'] == -0.05
    
    lens2 = LensModel.from_dict(data)
    assert lens2.k1 == 0.1
    assert lens2.k2 == -0.05

def test_mesh_model_reset():
    mesh = MeshModel(rows=3, cols=4)
    assert len(mesh.points) == 3
    assert len(mesh.points[0]) == 4
    # Corner check
    assert mesh.points[0][0].x == 0.0
    assert mesh.points[0][0].y == 0.0
    assert mesh.points[2][3].x == 1.0
    assert mesh.points[2][3].y == 1.0

def test_mesh_model_rotate_clockwise():
    # 2x3 grid:
    # (0,0) (0.5,0) (1,0)
    # (0,1) (0.5,1) (1,1)
    mesh = MeshModel(rows=2, cols=3)
    mesh.rotate_clockwise()
    
    # Becomes 3x2 grid
    assert mesh.rows == 3
    assert mesh.cols == 2
    
    # Original (0,0) [Top-Left] -> New (1,0) [Top-Right]?
    # No, wait. rotate_clockwise: temp_points[c][(self.rows - 1) - r] = self.points[r][c]
    # For (r=0, c=0) -> temp_points[0][(2-1)-0] = temp_points[0][1]
    # For (r=0, c=2) [Top-Right] -> temp_points[2][(2-1)-0] = temp_points[2][1]
    # Let's verify coordinate logic.
    assert mesh.points[0][1].x == 0.0 # Original (0,0)
    assert mesh.points[0][1].y == 0.0
    
def test_mesh_model_persistence():
    mesh = MeshModel(rows=2, cols=2)
    mesh.points[0][1].x = 0.9
    data = mesh.to_dict()
    assert data['rows'] == 2
    assert data['points'][0][1]['x'] == 0.9
    
    mesh2 = MeshModel.from_dict(data)
    assert mesh2.rows == 2
    assert mesh2.points[0][1].x == 0.9
