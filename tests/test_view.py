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
    assert win.current_pixmap is None
    # 縮小リサイズを許可する設定になっているか確認
    assert win.label.minimumSize() == QSize(1, 1)
    assert win.label.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
    assert win.label.sizePolicy().verticalPolicy() == QSizePolicy.Ignored

def test_preview_window_set_image(qapp):
    win = PreviewWindow()
    img = QImage(200, 100, QImage.Format_RGB888)
    img.fill(Qt.white)
    
    win.set_image(img)
    assert win.current_pixmap is not None
    assert win.current_pixmap.width() == 200
    assert win.current_pixmap.height() == 100
    
    # ラベルにPixmapがセットされているか
    assert not win.label.pixmap().isNull()

def test_preview_window_resize_scaling(qapp):
    win = PreviewWindow()
    # 200x200の画像をセット
    img = QImage(200, 200, QImage.Format_RGB888)
    img.fill(Qt.red)
    win.set_image(img)
    
    # ウィンドウ（ラベル）を 100x100 にリサイズ
    win.label.resize(100, 100)
    win.update_display()
    
    # ラベル内のPixmapサイズが 100x100 にスケールされているか確認
    # (Qt.KeepAspectRatio なので 100x100 になるはず)
    pixmap_size = win.label.pixmap().size()
    assert pixmap_size.width() == 100
    assert pixmap_size.height() == 100

    # さらに小さく 50x50 にリサイズ
    win.label.resize(50, 50)
    win.update_display()
    assert win.label.pixmap().size() == QSize(50, 50)
