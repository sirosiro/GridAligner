import sys
import os
import pytest
import json
import numpy as np
from unittest.mock import MagicMock, patch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import MeshModel, LensModel, Point
from controller import Controller
from PySide6.QtCore import QPointF

@pytest.fixture
def mock_view():
    view = MagicMock()
    # Mock some methods used in Controller
    view.get_grid_dimensions.return_value = (3, 3)
    view.get_lens_params.return_value = (0.1, -0.05, 0.01) # k1, k2, k3
    view.is_smooth_warp_enabled.return_value = False
    view.is_super_res_enabled.return_value = False
    
    # 子コンポーネントを単純なモックとして定義
    view.canvas = MagicMock()
    view.preview_window = MagicMock()
    view.check_show_crosshair = MagicMock()
    view.check_show_crosshair.isChecked.return_value = True
    
    # QMessageBox, QFileDialog の静的メソッドをグローバルにパッチ
    with patch('PySide6.QtWidgets.QMessageBox.information'), \
         patch('PySide6.QtWidgets.QMessageBox.warning'), \
         patch('PySide6.QtWidgets.QMessageBox.critical'), \
         patch('PySide6.QtWidgets.QMessageBox.question'):
        yield view

def test_controller_initialization(mock_view):
    mesh = MeshModel(rows=3, cols=3)
    lens = LensModel()
    ctrl = Controller(mock_view, mesh, lens)
    assert ctrl.mesh == mesh
    assert ctrl.lens == lens

def test_controller_lens_params_sync(mock_view):
    mesh = MeshModel(rows=3, cols=3)
    lens = LensModel(k1=0.0)
    ctrl = Controller(mock_view, mesh, lens)
    ctrl.original_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 物理座標同期を伴うレンズパラメータ更新
    ctrl._update_lens_params_with_sync(0.1, 0.2, 0.05)
    
    assert ctrl.lens.k1 == 0.1
    assert ctrl.lens.k2 == 0.2
    assert ctrl.lens.k3 == 0.05

def test_controller_sync_crosshair(mock_view):
    mesh = MeshModel(rows=3, cols=3)
    lens = LensModel()
    ctrl = Controller(mock_view, mesh, lens)
    
    # プレビュー窓を強制的にインスタンス化
    ctrl.preview_window = MagicMock()
    ctrl.preview_window.isVisible.return_value = True
    
    # メインキャンバスからの同期
    ctrl.sync_crosshair(0.7, 0.3, source=mock_view.canvas)
    ctrl.preview_window.set_crosshair_pos.assert_called()
    
    # プレビュー窓からの同期
    ctrl.sync_crosshair(0.2, 0.8, source=ctrl.preview_window)
    assert mock_view.canvas.update.called

def test_controller_guide_scale_changed(mock_view):
    mesh = MeshModel(rows=3, cols=3)
    lens = LensModel()
    ctrl = Controller(mock_view, mesh, lens)
    
    # プレビュー窓を強制的にインスタンス化
    ctrl.preview_window = MagicMock()
    ctrl.preview_window.isVisible.return_value = True
    
    # ガイドスケール変更の同期
    ctrl.on_guide_scale_changed(0.15)
    mock_view.canvas.set_guide_scale.assert_called_with(0.15)
    ctrl.preview_window.set_guide_scale.assert_called_with(0.15)

def test_controller_grid_operations(mock_view):
    mesh = MeshModel(rows=3, cols=3)
    lens = LensModel()
    ctrl = Controller(mock_view, mesh, lens)
    
    # 回転
    ctrl.rotate_grid()
    assert mesh.rows == 3 # 3x3 なので変わらない
    
    # 倍密化
    ctrl.subdivide_grid()
    assert mesh.rows == 5 # (3-1)*2 + 1
    
    # リセット (mock_view.get_grid_dimensions が (3, 3) を返す)
    ctrl.reset_mesh()
    assert mesh.rows == 3

def test_controller_save_load_project(mock_view, tmp_path):
    mesh = MeshModel(rows=2, cols=2)
    lens = LensModel(k1=0.2)
    ctrl = Controller(mock_view, mesh, lens)
    
    project_file = str(tmp_path / "project.json")
    
    # QFileDialog をモックして自動でパスを返すようにする
    with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName', return_value=(project_file, "")), \
         patch('PySide6.QtWidgets.QFileDialog.getOpenFileName', return_value=(project_file, "")):
        
        ctrl.save_project()
        assert os.path.exists(project_file)
        
        # 読み込み
        ctrl.load_project()
        assert ctrl.lens.k1 == 0.2
        assert ctrl.mesh.rows == 2

def test_controller_export(mock_view, tmp_path):
    mesh = MeshModel(rows=2, cols=2)
    lens = LensModel()
    ctrl = Controller(mock_view, mesh, lens)
    ctrl.original_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    export_file = str(tmp_path / "output.png")
    with patch('PySide6.QtWidgets.QFileDialog.getSaveFileName', return_value=(export_file, "")):
        ctrl.export()
    
    assert os.path.exists(export_file)
