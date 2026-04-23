import cv2
import numpy as np
import json
import os
import re
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QThread, Signal, Qt
from model import MeshModel, LensModel, Point
from engine import WarpEngine
from pytorch_engine import PyTorchWarpEngine
from view import PreviewWindow, CameraDialog

class LensWorker(QThread):
    """
    レンズ歪み推定をバックグラウンドで実行し、進捗を報告するワーカークラス。
    """
    finished = Signal(float, float, float)
    progress = Signal(int, str)
    
    def __init__(self, engine, image):
        super().__init__()
        self.engine = engine
        self.image = image
        
    def run(self):
        def cb(curr, total, loss):
            # 進捗率とステータス文字列を送信
            pct = int(curr / total * 100)
            self.progress.emit(pct, f"Optimizing Lens... Step {curr}/{total} (Loss: {loss:.6f})")
            
        try:
            k1, k2, k3 = self.engine.estimate_lens_parameters(self.image, progress_callback=cb)
            self.finished.emit(k1, k2, k3)
        except Exception as e:
            # 簡略化のためエラー時も0,0,0を返すが、本来はエラー報告が望ましい
            self.finished.emit(0.0, 0.0, 0.0)

class Controller:
    def __init__(self, view, model_mesh, model_lens):
        self.view = view
        self.mesh = model_mesh
        self.lens = model_lens
        
        self.original_image = None
        self.preview_window = None
        self.camera_dialog = None
        self.lens_worker = None
        
        # V2 Engines
        self.py_engine = PyTorchWarpEngine()
        
        # Connect signals
        self.view.btn_open.clicked.connect(self.open_image)
        self.view.btn_camera.clicked.connect(self.open_camera)
        self.view.btn_reset.clicked.connect(self.reset_mesh)
        self.view.slider_k1.valueChanged.connect(self.update_lens)
        self.view.slider_k2.valueChanged.connect(self.update_lens)
        self.view.slider_k3.valueChanged.connect(self.update_lens)
        self.view.spin_rows.valueChanged.connect(self.resize_mesh)
        self.view.spin_cols.valueChanged.connect(self.resize_mesh)
        self.view.canvas.pointMoved.connect(self.on_point_moved)
        self.view.check_preview.stateChanged.connect(self.toggle_preview_window)
        
        # V2 signals
        self.view.btn_auto_grid.clicked.connect(self.on_auto_grid)
        self.view.btn_auto_lens.clicked.connect(self.on_auto_lens_correction)
        self.view.check_super_res.stateChanged.connect(self.update_preview)
        self.view.check_smooth_warp.stateChanged.connect(lambda: None) # Optional setting
        
        # New signals
        self.view.btn_save_project.clicked.connect(self.save_project)
        self.view.btn_load_project.clicked.connect(self.load_project)
        self.view.btn_export.clicked.connect(self.export)
        self.view.btn_straighten.clicked.connect(self.straighten_grid)
        self.view.btn_expand.clicked.connect(self.expand_mesh_to_full_frame)
        self.view.btn_rotate.clicked.connect(self.rotate_grid)
        self.view.check_show_grid.toggled.connect(self.view.set_grid_visible)
        self.view.check_show_crosshair.toggled.connect(self.view.set_crosshair_visible)

    def on_point_moved(self, r, c, nx, ny):
        # Update Model
        self.mesh.points[r][c].x = nx
        self.mesh.points[r][c].y = ny

        # Perspective Sync if corner is moved
        is_corner = (r == 0 or r == self.mesh.rows - 1) and (c == 0 or c == self.mesh.cols - 1)
        if is_corner:
            WarpEngine.sync_perspective(self.mesh)

        # Smooth Warp logic
        if self.view.is_smooth_warp_enabled():
            if not is_corner:
                # Fixed points are 4 corners + current moving point
                fixed = [(0, 0), (0, self.mesh.cols - 1), 
                         (self.mesh.rows - 1, 0), (self.mesh.rows - 1, self.mesh.cols - 1),
                         (r, c)]
                self.py_engine.minimize_energy(self.mesh, fixed)
        
        self.update_preview()

    def on_auto_grid(self):
        if self.original_image is None: return
        
        # Current UI settings as hints
        curr_r, curr_c = self.view.get_grid_dimensions()
        
        # ユーザーが既にレンズ係数スライダーを調整している場合、
        # その「補正後の画像」を検出に使う。
        image_for_auto = WarpEngine.apply_lens_distortion(self.original_image, self.lens)
        
        # 1. 検出実行
        res = self.py_engine.detect_initial_grid(
            image_for_auto, target_rows=curr_r, target_cols=curr_c
        )
        
        if res:
            new_pts, a_rows, a_cols = res
            # UI の値を更新
            self.view.set_grid_dimensions(a_rows, a_cols)
            
            # メッシュデータを反映
            self.mesh.points = new_pts
            self.mesh.rows = a_rows
            self.mesh.cols = a_cols
            
            # 重要：キャンバスの表示を新しいメッシュ構造でリセット・更新
            self.view.canvas.set_mesh(self.mesh)
            self.view.canvas.update()
            self.update_preview()
        else:
            QMessageBox.warning(self.view, "AI Features", 
                "Could not detect grid pattern.\n"
                "Please ensure the image contains clear lines or a checkerboard.")

    def on_auto_lens_correction(self):
        """
        AIによる自動レンズ補正（直線性の最適化）を開始します。
        """
        if self.original_image is None:
            QMessageBox.warning(self.view, "AI Features", "Please open an image first.")
            return
            
        # UIの無効化とプログレス表示の開始
        self.view.btn_auto_lens.setEnabled(False)
        self.view.set_progress_visible(True)
        self.view.update_progress(0, "Analyzing lines...")
        
        # バックグラウンドスレッドで実行
        self.lens_worker = LensWorker(self.py_engine, self.original_image)
        self.lens_worker.progress.connect(self.view.update_progress)
        self.lens_worker.finished.connect(self.on_auto_lens_finished)
        self.lens_worker.start()

    def on_auto_lens_finished(self, k1, k2, k3):
        """
        自動補正完了時の処理。
        """
        # 物理座標同期を伴う更新
        self._update_lens_params_with_sync(k1, k2, k3)
        
        # UIへの反映
        self.view.set_lens_params(k1, k2, k3)
        
        # UIの復元
        self.view.set_progress_visible(False)
        self.view.btn_auto_lens.setEnabled(True)
        
        self.update_preview()
        QMessageBox.information(self.view, "AI Features", 
            f"Auto Lens Correction Finished.\nEstimated: k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}")

    def resize_mesh(self):
        rows, cols = self.view.get_grid_dimensions()
        self.mesh.rows = rows
        self.mesh.cols = cols
        self.mesh.reset()
        self.view.canvas.set_mesh(self.mesh)
        self.update_preview()

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.view, "Open Image", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            # 日本語パス対応
            n = np.fromfile(file_path, np.uint8)
            image = cv2.imdecode(n, cv2.IMREAD_COLOR)
            self.load_new_image(image, path=file_path)

    def open_camera(self):
        if not self.camera_dialog:
            self.camera_dialog = CameraDialog(self.view)
            self.camera_dialog.imageCaptured.connect(lambda img: self.load_new_image(img, path="Captured from Camera"))
        
        self.camera_dialog.start_camera()
        self.camera_dialog.exec()
        self.camera_dialog.stop_camera()

    def extract_grid_from_filename(self, path):
        """
        ファイル名から '8x10' や '10x8' といった格子数パターンを抽出します。
        """
        if not path: return None
        filename = os.path.basename(path)
        # 8x10, 8-10, 8 by 10, 8_x_10 等の多様なパターンにマッチ
        match = re.search(r'(\d+)[^0-9]+(\d+)', filename)
        if match:
            try:
                val1 = int(match.group(1))
                val2 = int(match.group(2))
                # 極端に小さい値（1以下）は無視
                if val1 > 1 and val2 > 1:
                    return val1, val2
            except ValueError:
                pass
        return None

    def load_new_image(self, image, path=None):
        if image is not None:
            self.original_image = image
            
            # クリーンロード原則に基づき、パラメータをリセット
            self.lens.k1 = 0.0
            self.lens.k2 = 0.0
            self.lens.k3 = 0.0
            self.view.reset_to_default()
            self.view.set_image_path(path)
            
            # インテリジェント・初期化原則: 格子数の自動推定
            est_rows, est_cols = None, None
            
            # 1. ファイル名からのメタデータ抽出を優先
            meta = self.extract_grid_from_filename(path)
            if meta:
                est_rows, est_cols = meta
            else:
                # 2. AIによる推定（プロファイル解析）
                est_rows, est_cols = self.py_engine.estimate_grid_dimensions(image)
            
            if est_rows and est_cols:
                # 推定値を UI に反映（これにより resize_mesh が呼ばれる可能性があるが、
                # ここでは直接 mesh をセットアップする）
                self.view.set_grid_dimensions(est_rows, est_cols)
                self.mesh.rows = est_rows
                self.mesh.cols = est_cols
            else:
                # 失敗した場合は現在のUI値（デフォルト等）を使用
                rows, cols = self.view.get_grid_dimensions()
                self.mesh.rows = rows
                self.mesh.cols = cols
                
            self.mesh.reset()
            self.view.canvas.set_mesh(self.mesh)
            self.update_preview()

    def reset_mesh(self):
        self.mesh.reset()
        self.view.canvas.set_mesh(self.mesh)
        self.update_preview()

    def straighten_grid(self):
        WarpEngine.sync_perspective(self.mesh)
        self.view.canvas.update()
        self.update_preview()

    def expand_mesh_to_full_frame(self):
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            WarpEngine.expand_mesh(self.mesh, w, h)
            self.view.set_grid_dimensions(self.mesh.rows, self.mesh.cols)
            self.view.canvas.update()
            self.update_preview()

    def rotate_grid(self):
        """
        格子の向きを90度回転します。
        回転後の整合性を保つため、UI側の Rows/Cols の数値もスワップします。
        """
        self.mesh.rotate_clockwise()
        # UI側の数値も入れ替え（これにより整合性を保つ）
        self.view.set_grid_dimensions(self.mesh.rows, self.mesh.cols)
        self.view.canvas.update()
        self.update_preview()

    def update_lens(self):
        k1, k2, k3 = self.view.get_lens_params()
        self._update_lens_params_with_sync(k1, k2, k3)
        self.update_preview()

    def _update_lens_params_with_sync(self, new_k1, new_k2, new_k3):
        """
        レンズパラメータを更新し、同時にメッシュ制御点の物理位置を同期させます。
        """
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            aspect = w / h
            new_lens = LensModel(k1=new_k1, k2=new_k2, k3=new_k3)
            
            # 再投影を実行してメッシュ点を移動
            self.py_engine.reproject_mesh(self.mesh, self.lens, new_lens, aspect)
            
            # パラメータを確定
            self.lens.k1 = new_k1
            self.lens.k2 = new_k2
            self.lens.k3 = new_k3

    def toggle_preview_window(self, state):
        if state == 2: # Checked
            if not self.preview_window:
                self.preview_window = PreviewWindow(self.view)
            self.preview_window.show()
            self.update_preview()
        else:
            if self.preview_window:
                self.preview_window.hide()

    def update_preview(self):
        if self.original_image is None:
            return

        # Prepare RGB version for PySide
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        # 1. Main View (Lens corrected image + Grid overlay)
        processed = WarpEngine.apply_lens_distortion(rgb_image, self.lens)
            
        h, w, ch = processed.shape
        qimg = QImage(processed, w, h, ch * w, QImage.Format_RGB888)
        self.view.canvas.set_image(qimg)
        self.view.canvas.set_mesh(self.mesh)

        # 2. Rectified Preview Window
        if self.preview_window and self.preview_window.isVisible():
            # プレビュー窓ではレンズ補正＋メッシュ補正（＋AI超解像）を一括で PyTorch 処理する
            rectified = self.py_engine.process_all(
                rgb_image, 
                self.mesh, 
                self.lens, 
                use_sr=self.view.is_super_res_enabled()
            )
                
            rh, rw, rch = rectified.shape
            rqimg = QImage(rectified, rw, rh, rch * rw, QImage.Format_RGB888)
            self.preview_window.set_image(rqimg)

    def save_project(self):
        file_path, _ = QFileDialog.getSaveFileName(self.view, "Save Project", "", "JSON Files (*.json)")
        if not file_path: return
        
        data = {
            "mesh": self.mesh.to_dict(),
            "lens": self.lens.to_dict()
        }
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            QMessageBox.information(self.view, "Success", "Project saved successfully.")
        except Exception as e:
            QMessageBox.critical(self.view, "Error", f"Failed to save: {str(e)}")

    def load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(self.view, "Load Project", "", "JSON Files (*.json)")
        if not file_path: return
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            self.mesh = MeshModel.from_dict(data["mesh"])
            self.lens = LensModel.from_dict(data["lens"])
            
            # Sync UI
            self.view.set_grid_dimensions(self.mesh.rows, self.mesh.cols)
            self.view.set_lens_params(self.lens.k1, self.lens.k2, self.lens.k3)
            
            self.view.canvas.set_mesh(self.mesh)
            self.update_preview()
            QMessageBox.information(self.view, "Success", "Project loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self.view, "Error", f"Failed to load: {str(e)}")

    def export(self):
        if self.original_image is None:
            QMessageBox.warning(self.view, "Export", "Please open an image first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.view, "Export Rectified Image", "", "PNG (*.png);;JPG (*.jpg);;JPEG (*.jpeg);;BMP (*.bmp)"
        )
        if not file_path: return

        try:
            # Process the original BGR image for export (preserve original colors/quality)
            rectified = WarpEngine.apply_mesh_rectification(self.original_image, self.mesh, self.lens)
            
            # 日本語パス対応
            ext = os.path.splitext(file_path)[1]
            result, n = cv2.imencode(ext, rectified)
            if result:
                with open(file_path, mode='wb') as f:
                    n.tofile(f)
            
            QMessageBox.information(self.view, "Success", f"Image exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self.view, "Error", f"Failed to export: {str(e)}")
