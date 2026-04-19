import sys
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QSlider, QLabel, QFileDialog, QFrame, QCheckBox, QDialog, QSpinBox, QSizePolicy)
from PySide6.QtCore import Qt, QPointF, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QAction
import cv2
import numpy as np

class Canvas(QWidget):
    pointMoved = Signal(int, int, float, float) # (row, col, new_x, new_y)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.pixmap = None
        self.mesh = None
        self.selected_point = None
        self.constraint_mode = False # Alt key
        
        # 描画補助のフラグを追加。レンズ歪み調整時にグリッドが邪魔になるのを防ぎ、
        # 直線ガイドで補正精度を確認するために使用。
        self.show_grid = True
        self.show_crosshair = False
        self.crosshair_pos = QPointF(0.5, 0.5) # 正規化座標 (0-1)
        
        self.setMouseTracking(True)
        self.setMinimumSize(600, 400)

    def set_image(self, qimage):
        self.image = qimage
        self.pixmap = QPixmap.fromImage(qimage)
        self.update()

    def set_mesh(self, mesh):
        self.mesh = mesh
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.pixmap:
            # Draw image centered
            scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)
            
            self.draw_grid(painter, scaled_pixmap, x, y)
        else:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open an image to start")

    def draw_grid(self, painter, pixmap, offset_x, offset_y):
        if not self.mesh:
            return

        w, h = pixmap.width(), pixmap.height()

        # 1. 十字ガイドの描画 (レンズ歪みの直線チェック用)
        if self.show_crosshair:
            guide_pen = QPen(QColor(255, 255, 0, 180), 1, Qt.DashLine) # 黄色の破線
            painter.setPen(guide_pen)
            cx = offset_x + self.crosshair_pos.x() * w
            cy = offset_y + self.crosshair_pos.y() * h
            painter.drawLine(offset_x, cy, offset_x + w, cy) # 水平
            painter.drawLine(cx, offset_y, cx, offset_y + h) # 垂直

        # 2. グリッドと制御点の描画
        if self.show_grid:
            grid_pen = QPen(QColor(0, 255, 0, 150), 1)
            painter.setPen(grid_pen)

            # Draw lines
            for r in range(self.mesh.rows):
                for c in range(self.mesh.cols):
                    p = self.mesh.points[r][c]
                    px = offset_x + p.x * w
                    py = offset_y + p.y * h
                    
                    # Right line
                    if c < self.mesh.cols - 1:
                        p2 = self.mesh.points[r][c+1]
                        painter.drawLine(px, py, offset_x + p2.x * w, offset_y + p2.y * h)
                    # Bottom line
                    if r < self.mesh.rows - 1:
                        p2 = self.mesh.points[r+1][c]
                        painter.drawLine(px, py, offset_x + p2.x * w, offset_y + p2.y * h)

            # Draw points
            for r in range(self.mesh.rows):
                for c in range(self.mesh.cols):
                    p = self.mesh.points[r][c]
                    painter.setBrush(QBrush(QColor(255, 0, 0, 200)))
                    painter.drawEllipse(QPointF(offset_x + p.x * w, offset_y + p.y * h), 4, 4)

    def mousePressEvent(self, event):
        if not self.mesh or not self.pixmap:
            return
            
        scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        offset_x = (self.width() - scaled_pixmap.width()) // 2
        offset_y = (self.height() - scaled_pixmap.height()) // 2
        w, h = scaled_pixmap.width(), scaled_pixmap.height()

        for r in range(self.mesh.rows):
            for c in range(self.mesh.cols):
                p = self.mesh.points[r][c]
                screen_x = offset_x + p.x * w
                screen_y = offset_y + p.y * h
                if (event.position() - QPointF(screen_x, screen_y)).manhattanLength() < 10:
                    self.selected_point = (r, c)
                    return
        
        # 制御点以外をクリックした場合は、十字ガイドラインの位置を更新
        self.crosshair_pos = QPointF(
            (event.position().x() - offset_x) / w,
            (event.position().y() - offset_y) / h
        )
        self.update()

    def mouseMoveEvent(self, event):
        if self.selected_point and self.pixmap:
            scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            offset_x = (self.width() - scaled_pixmap.width()) // 2
            offset_y = (self.height() - scaled_pixmap.height()) // 2
            w, h = scaled_pixmap.width(), scaled_pixmap.height()

            r, c = self.selected_point
            new_nx = (event.position().x() - offset_x) / w
            new_ny = (event.position().y() - offset_y) / h
            
            new_nx = max(0.0, min(1.0, new_nx))
            new_ny = max(0.0, min(1.0, new_ny))

            if self.constraint_mode:
                old_p = self.mesh.points[r][c]
                dx = abs(new_nx - old_p.x)
                dy = abs(new_ny - old_p.y)
                if dx > dy:
                    new_ny = old_p.y
                else:
                    new_nx = old_p.x

            # Send event to controller instead of modifying model directly
            self.pointMoved.emit(r, c, new_nx, new_ny)
            self.update()

    def mouseReleaseEvent(self, event):
        self.selected_point = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Alt:
            self.constraint_mode = True

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Alt:
            self.constraint_mode = False

class CameraDialog(QDialog):
    imageCaptured = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Capture")
        self.setMinimumSize(640, 480)
        
        layout = QVBoxLayout(self)
        self.video_label = QLabel("Starting camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)

        self.btn_capture = QPushButton("📷 Take Snapshot")
        self.btn_capture.setMinimumHeight(40)
        self.btn_capture.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.btn_capture.clicked.connect(self.capture)
        layout.addWidget(self.btn_capture)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.last_frame = None

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            self.video_label.setText("Error: Camera not found.")
            return
            
        self.timer.start(33)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_frame(self):
        ret, frame = False, None
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
        if not ret:
            # Fallback mock
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "MOCK CAMERA", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret = True

        if ret:
            self.last_frame = frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qimg = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def capture(self):
        if self.last_frame is not None:
            self.imageCaptured.emit(self.last_frame.copy())
            self.accept()

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)

class PreviewWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rectified Preview")
        
        self.label = QLabel("No image loaded")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #222; color: #888;")
        
        # 縮小リサイズを可能にするための設定
        # QLabelはデフォルトでpixmapサイズに固執するため、最小サイズを解除する
        self.label.setMinimumSize(1, 1)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.setCentralWidget(self.label)
        self.resize(600, 450)
        self.current_pixmap = None

    def set_image(self, qimage):
        if qimage.isNull(): return
        self.current_pixmap = QPixmap.fromImage(qimage)
        self.update_display()

    def update_display(self):
        if self.current_pixmap:
            # ラベルの現在の（レイアウトによって決定された）サイズに合わせてスケーリング
            self.label.setPixmap(self.current_pixmap.scaled(
                self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GridAligner")
        self.resize(1100, 750)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Left: Canvas + Path Info
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = Canvas()
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        left_layout.addWidget(self.canvas, stretch=1)
        
        self.lbl_path = QLabel("No image loaded")
        self.lbl_path.setWordWrap(True)
        self.lbl_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_path.setStyleSheet("color: #aaa; background-color: #1a1a1a; padding: 4px; border-top: 1px solid #333;")
        left_layout.addWidget(self.lbl_path)
        
        layout.addWidget(left_panel, stretch=4)

        # Right: Controls
        controls = QWidget()
        controls.setFixedWidth(250)
        controls_layout = QVBoxLayout(controls)
        layout.addWidget(controls)

        self.btn_open = QPushButton("📁 Open Image File")
        controls_layout.addWidget(self.btn_open)

        self.btn_camera = QPushButton("🎥 Get from Camera")
        self.btn_camera.setStyleSheet("background-color: #2196F3; color: white;")
        controls_layout.addWidget(self.btn_camera)

        controls_layout.addSpacing(10)
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
        controls_layout.addWidget(line)
        controls_layout.addWidget(QLabel("<b>AI Features (V2)</b>"))

        self.btn_auto_grid = QPushButton("✨ Auto-Grid Detection")
        self.btn_auto_grid.setStyleSheet("background-color: #673AB7; color: white;")
        controls_layout.addWidget(self.btn_auto_grid)

        self.check_smooth_warp = QCheckBox("Smooth Warp (Physics)")
        self.check_smooth_warp.setChecked(True)
        controls_layout.addWidget(self.check_smooth_warp)

        self.check_super_res = QCheckBox("Neural Super-Res (SR)")
        self.check_super_res.setStyleSheet("color: #E91E63;")
        controls_layout.addWidget(self.check_super_res)

        controls_layout.addSpacing(10)
        line_old = QFrame(); line_old.setFrameShape(QFrame.HLine); line_old.setFrameShadow(QFrame.Sunken)
        controls_layout.addWidget(line_old)
        controls_layout.addWidget(QLabel("<b>Grid Settings</b>"))

        mesh_ctrl = QHBoxLayout()
        mesh_ctrl.addWidget(QLabel("Rows:"))
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(2, 20); self.spin_rows.setValue(5)
        mesh_ctrl.addWidget(self.spin_rows)
        mesh_ctrl.addWidget(QLabel("Cols:"))
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(2, 20); self.spin_cols.setValue(5)
        mesh_ctrl.addWidget(self.spin_cols)
        controls_layout.addLayout(mesh_ctrl)

        self.btn_reset = QPushButton("Reset Mesh")
        controls_layout.addWidget(self.btn_reset)

        self.btn_straighten = QPushButton("📏 Straighten (Align to Corners)")
        self.btn_straighten.setToolTip("Align all internal points to the 4 corners linearly.")
        controls_layout.addWidget(self.btn_straighten)

        self.btn_expand = QPushButton("🚀 Expand to Full Frame")
        self.btn_expand.setToolTip("Expand the current perspective plane to cover the whole image.")
        self.btn_expand.setStyleSheet("background-color: #FF9800; color: white;")
        controls_layout.addWidget(self.btn_expand)

        self.btn_rotate = QPushButton("↻ Rotate Grid (90° CW)")
        self.btn_rotate.setToolTip("Rotate the grid orientation by 90 degrees clockwise.")
        controls_layout.addWidget(self.btn_rotate)

        k1_layout = QHBoxLayout()
        k1_layout.addWidget(QLabel("k1 (Barrel/Pincushion)"))
        self.label_k1 = QLabel("0.00")
        self.label_k1.setAlignment(Qt.AlignRight)
        k1_layout.addWidget(self.label_k1)
        controls_layout.addLayout(k1_layout)

        self.slider_k1 = QSlider(Qt.Horizontal)
        self.slider_k1.setRange(-200, 200); self.slider_k1.setValue(0)
        controls_layout.addWidget(self.slider_k1)
        self.slider_k1.valueChanged.connect(self._update_labels)

        k2_layout = QHBoxLayout()
        k2_layout.addWidget(QLabel("k2 (Higher Order)"))
        self.label_k2 = QLabel("0.000")
        self.label_k2.setAlignment(Qt.AlignRight)
        k2_layout.addWidget(self.label_k2)
        controls_layout.addLayout(k2_layout)

        self.slider_k2 = QSlider(Qt.Horizontal)
        self.slider_k2.setRange(-200, 200); self.slider_k2.setValue(0)
        controls_layout.addWidget(self.slider_k2)
        self.slider_k2.valueChanged.connect(self._update_labels)

        self.btn_reset_lens = QPushButton("Reset Lens Parameters")
        controls_layout.addWidget(self.btn_reset_lens)
        self.btn_reset_lens.clicked.connect(self.reset_lens_sliders)

        line2 = QFrame(); line2.setFrameShape(QFrame.HLine); line2.setFrameShadow(QFrame.Sunken)
        controls_layout.addWidget(line2)
        controls_layout.addSpacing(10)

        self.check_preview = QCheckBox("Show Rectified Preview")
        self.check_preview.setStyleSheet("font-weight: bold; color: #4CAF50;")
        controls_layout.addWidget(self.check_preview)

        # 描画補助用チェックボックスを追加 (レンズ歪み調整時の視認性向上)
        self.check_show_grid = QCheckBox("Show Grid Lines")
        self.check_show_grid.setChecked(True)
        controls_layout.addWidget(self.check_show_grid)

        self.check_show_crosshair = QCheckBox("Show Crosshair Guide")
        self.check_show_crosshair.setChecked(False)
        controls_layout.addWidget(self.check_show_crosshair)

        controls_layout.addSpacing(10)
        controls_layout.addWidget(QLabel("<b>Projects & Export</b>"))
        
        self.btn_save_project = QPushButton("💾 Save Project (.json)")
        controls_layout.addWidget(self.btn_save_project)
        
        self.btn_load_project = QPushButton("📂 Load Project (.json)")
        controls_layout.addWidget(self.btn_load_project)

        self.btn_export = QPushButton("🚀 Export Rectified Image")
        self.btn_export.setMinimumHeight(40)
        self.btn_export.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        controls_layout.addWidget(self.btn_export)

        controls_layout.addStretch()

    # --- UI Abstraction Methods ---
    def get_grid_dimensions(self):
        return self.spin_rows.value(), self.spin_cols.value()

    def set_grid_dimensions(self, r, c):
        self.spin_rows.blockSignals(True)
        self.spin_cols.blockSignals(True)
        self.spin_rows.setValue(r)
        self.spin_cols.setValue(c)
        self.spin_rows.blockSignals(False)
        self.spin_cols.blockSignals(False)

    def get_lens_params(self):
        # Return scaled values
        return self.slider_k1.value() / 100.0, self.slider_k2.value() / 500.0

    def set_lens_params(self, k1, k2):
        self.slider_k1.setValue(int(k1 * 100))
        self.slider_k2.setValue(int(k2 * 500))

    def is_preview_enabled(self):
        return self.check_preview.isChecked()

    def is_super_res_enabled(self):
        return self.check_super_res.isChecked()

    def is_smooth_warp_enabled(self):
        return self.check_smooth_warp.isChecked()

    def set_grid_visible(self, visible):
        self.canvas.show_grid = visible
        self.canvas.update()

    def set_crosshair_visible(self, visible):
        self.canvas.show_crosshair = visible
        self.canvas.update()

    def _update_labels(self):
        k1, k2 = self.get_lens_params()
        self.label_k1.setText(f"{k1:.2f}")
        self.label_k2.setText(f"{k2:.3f}")

    def reset_lens_sliders(self):
        self.slider_k1.setValue(0)
        self.slider_k2.setValue(0)
        self._update_labels()

    def reset_to_default(self):
        """全ての補正パラメータを初期状態にリセットします。"""
        self.set_grid_dimensions(9, 9)
        self.reset_lens_sliders()

    def set_image_path(self, path):
        """画像パスの表示を更新します。ファイル名を強調表示します。"""
        if not path:
            self.lbl_path.setText("No image loaded")
            return
            
        import os
        basename = os.path.basename(path)
        # HTML形式でファイル名を強調
        self.lbl_path.setText(f"<b>File: {basename}</b><br><small>{path}</small>")
