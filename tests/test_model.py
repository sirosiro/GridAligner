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
    lens = LensModel(k1=0.1, k2=-0.05, k3=0.01)
    data = lens.to_dict()
    assert data['k1'] == 0.1
    assert data['k2'] == -0.05
    assert data['k3'] == 0.01
    
    lens2 = LensModel.from_dict(data)
    assert lens2.k1 == 0.1
    assert lens2.k2 == -0.05
    assert lens2.k3 == 0.01

def test_mesh_model_reset():
    mesh = MeshModel(rows=3, cols=4)
    assert len(mesh.points) == 3
    assert len(mesh.points[0]) == 4
    # Corner check
    assert mesh.points[0][0].x == 0.0
    assert mesh.points[0][0].y == 0.0
    assert mesh.points[2][3].x == 1.0
    assert mesh.points[2][3].y == 1.0

def test_mesh_model_subdivide():
    mesh = MeshModel(rows=3, cols=3) # 3x3 cells (4 points per side)
    mesh.subdivide()
    # 3 cells becomes 6 cells -> 7 points
    # New rows = (3-1)*2 + 1 = 5
    assert mesh.rows == 5
    assert mesh.cols == 5
    assert mesh.points[0][0].x == 0.0
    assert mesh.points[4][4].x == 1.0

def test_mesh_model_transform_by_corners():
    mesh = MeshModel(rows=2, cols=2)
    # Target corners (Clockwise): TL, TR, BR, BL
    new_corners = [
        Point(0.1, 0.1), Point(0.9, 0.1),
        Point(0.9, 0.9), Point(0.1, 0.9)
    ]
    # Linearize=True should move points to exactly match homography of corners
    mesh.transform_by_corners(new_corners, linearize=True)
    assert abs(mesh.points[0][0].x - 0.1) < 1e-6
    assert abs(mesh.points[1][1].x - 0.9) < 1e-6

def test_mesh_model_rotate_clockwise():
    mesh = MeshModel(rows=2, cols=3)
    mesh.rotate_clockwise()
    assert mesh.rows == 3
    assert mesh.cols == 2
    assert mesh.points[0][1].x == 0.0
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
