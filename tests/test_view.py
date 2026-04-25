import sys
import os
import pytest
from PySide6.QtWidgets import QApplication, QSizePolicy
from PySide6.QtGui import QImage
from PySide6.QtCore import Qt, QSize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from view import PreviewWindow

# GUIテスト用のQApplicationインスタンス
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

def test_preview_window_initialization(qapp):
    win = PreviewWindow()
    assert win.windowTitle() == "Rectified Preview"
    # PreviewCanvasがセットされているか確認
    assert hasattr(win, "canvas")
    # 縮小リサイズを許可する設定になっているか確認
    assert win.canvas.minimumSize() == QSize(1, 1)
    # Policyの比較
    assert win.canvas.sizePolicy().horizontalPolicy() == QSizePolicy.Preferred
    assert win.canvas.sizePolicy().verticalPolicy() == QSizePolicy.Preferred

def test_preview_window_set_image(qapp):
    win = PreviewWindow()
    img = QImage(200, 100, QImage.Format_RGB888)
    img.fill(Qt.white)
    
    win.set_image(img)
    assert win.canvas.pixmap is not None
    assert win.canvas.pixmap.width() == 200
    assert win.canvas.pixmap.height() == 100
    
    # CanvasにPixmapが保持されているか
    assert not win.canvas.pixmap.isNull()

def test_preview_window_resize_scaling(qapp):
    win = PreviewWindow()
    # 200x200の画像をセット
    img = QImage(200, 200, QImage.Format_RGB888)
    img.fill(Qt.red)
    win.set_image(img)
    
    # ウィンドウのリサイズに応じて描画が更新されるか
    win.resize(100, 100)
    win.update()
    # (描画結果のPixmapサイズの厳密なチェックは QPainter/Canvas の内部挙動に依存するため、ここではクラッシュしないことを確認)

    # ガイドスケールスライダーが存在し、範囲が正しいか
    assert win.slider_guide_scale.minimum() == 100
    assert win.slider_guide_scale.maximum() == 2000
