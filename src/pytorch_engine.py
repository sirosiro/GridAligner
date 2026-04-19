import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import re
from model import MeshModel, LensModel, Point

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

class SuperResModel(nn.Module):
    def __init__(self, upscale_factor=2):
        super(SuperResModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3 * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

class PyTorchWarpEngine:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cpu")
        self.sr_model = SuperResModel(upscale_factor=2).to(self.device).eval()

    def get_target_size(self, src_w, src_h, mesh: MeshModel):
        dst_aspect = (mesh.cols - 1) / (mesh.rows - 1) if mesh.rows > 1 else 1.0
        if dst_aspect > 1.0:
            return src_w, int(src_w / dst_aspect)
        return int(src_h * dst_aspect), src_h

    def get_lens_grid(self, w, h, lens: LensModel):
        yy, xx = torch.meshgrid(torch.linspace(-0.5 * (h/w), 0.5 * (h/w), h, device=self.device), 
                                torch.linspace(-0.5, 0.5, w, device=self.device), indexing='ij')
        r2 = xx**2 + yy**2
        dist = 1 + lens.k1 * r2 + lens.k2 * (r2**2)
        return torch.stack((xx * 2 * dist, yy * 2 * dist * (w/h)), dim=-1).unsqueeze(0)

    def _find_grid_lines(self, hist, size, expected=None):
        """
        ヒストグラムからピーク（格子の線）を検出します。
        expected が指定されている場合は、その数に近い結果が得られるよう調整します。
        expected が None の場合は、自律的にピーク数をカウントします。
        """
        sigma = 21 if expected and expected > 10 else 51
        blurred = cv2.GaussianBlur(hist.astype(np.float32), (sigma, 1), 0).flatten()
        
        peaks = []
        # expected がない場合は、ヒストグラムの解像度から最小ギャップを推定
        min_gap = size / (expected + 2) if expected else size * 0.03
        thresh = np.mean(blurred) * 1.1
        
        for i in range(1, len(blurred)-1):
            if blurred[i] > blurred[i-1] and blurred[i] > blurred[i+1] and blurred[i] > thresh:
                if not peaks or abs(i - peaks[-1]) > min_gap:
                    peaks.append(i)
        
        if not peaks:
            return [size * (i/(expected-1)) for i in range(expected)] if expected else [size*0.1, size*0.9]
            
        # 期待値がある場合の補完ロジック
        if expected and len(peaks) != expected:
            gap = np.median(np.diff(peaks)) if len(peaks) > 1 else size * 0.1
            while len(peaks) < expected and peaks[0] > gap * 1.5:
                peaks.insert(0, peaks[0] - gap)
            while len(peaks) < expected and peaks[-1] < size - gap * 1.5:
                peaks.append(peaks[-1] + gap)
        
        return peaks

    def estimate_grid_dimensions(self, image_np):
        """
        AI（プロファイル解析）を用いて画像の格子数を推定します。
        """
        h, w = image_np.shape[:2]
        # Cannyエッジ検出の合成
        edges = np.zeros((h, w), dtype=np.uint8)
        for i in range(3):
            edges = cv2.bitwise_or(edges, cv2.Canny(image_np[:, :, i], 50, 150))
        
        y_lines = self._find_grid_lines(np.sum(edges, axis=1), h, expected=None)
        x_lines = self._find_grid_lines(np.sum(edges, axis=0), w, expected=None)
        
        # 検出されたピーク数を格子数（rows, cols）として返す
        return len(y_lines), len(x_lines)

    def detect_initial_grid(self, image_np, target_rows=None, target_cols=None):
        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        # --- Tier 1: Checkerboard Pattern (OpenCV SB/Legacy) ---
        if target_rows and target_cols:
            candidates = [(target_cols-2, target_rows-2), (target_cols-1, target_rows-1), (target_cols, target_rows)]
            for p_cols, p_rows in candidates:
                if p_cols <= 1 or p_rows <= 1: continue
                try:
                    ret = False
                    if p_cols > 2 and p_rows > 2 and hasattr(cv2, 'findChessboardCornersSB'):
                        ret, corners = cv2.findChessboardCornersSB(gray, (p_cols, p_rows), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
                    if not ret:
                        ret, corners = cv2.findChessboardCorners(gray, (p_cols, p_rows), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
                    
                    if ret:
                        if not hasattr(cv2, 'findChessboardCornersSB') or len(corners.shape) == 3:
                            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                        
                        M, _ = cv2.findHomography(np.array([[c, r] for r in range(p_rows) for c in range(p_cols)], dtype=np.float32), corners.reshape(-1, 2))
                        if M is not None:
                            res_pts = []
                            off_r, off_c = (target_rows - p_rows) / 2.0, (target_cols - p_cols) / 2.0
                            for r in range(target_rows):
                                row = []
                                for c in range(target_cols):
                                    pt = cv2.perspectiveTransform(np.array([[[c - off_c, r - off_r]]], dtype=np.float32), M)[0][0]
                                    row.append(Point(float(max(0, min(w, pt[0]))/w), float(max(0, min(h, pt[1]))/h)))
                                res_pts.append(row)
                            return res_pts, target_rows, target_cols
                except: continue

        # --- Tier 2: Statistical Profile Analysis ---
        edges = np.zeros((h, w), dtype=np.uint8)
        for i in range(3):
            edges = cv2.bitwise_or(edges, cv2.Canny(image_np[:, :, i], 50, 150))
        
        y_lines = self._find_grid_lines(np.sum(edges, axis=1), h, target_rows)
        x_lines = self._find_grid_lines(np.sum(edges, axis=0), w, target_cols)
        rows, cols = len(y_lines), len(x_lines)
        
        crude = np.array([[x, y] for y in y_lines for x in x_lines], dtype=np.float32).reshape(-1, 1, 2)
        try:
            refined = cv2.cornerSubPix(edges, crude, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            refined = refined.reshape(rows, cols, 2)
        except:
            refined = crude.reshape(rows, cols, 2)
        
        res_pts = []
        for r in range(rows):
            row = []
            for c in range(cols):
                px, py = refined[r, c]
                row.append(Point(float(max(0, min(w, px))/w), float(max(0, min(h, py))/h)))
            res_pts.append(row)
        return res_pts, rows, cols

    def process_all(self, image_np, mesh: MeshModel, lens: LensModel, use_sr: bool):
        src_h, src_w = image_np.shape[:2]
        dst_w, dst_h = self.get_target_size(src_w, src_h, mesh)
        img_t = torch.from_numpy(np.ascontiguousarray(image_np.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device)/255.0
        
        # Consistent Map Composition Logic
        rows, cols = mesh.rows, mesh.cols
        sparse_g = torch.stack([torch.tensor([mesh.points[r][c].x*2-1, mesh.points[r][c].y*2-1]) for r in range(rows) for c in range(cols)]).view(1, rows, cols, 2).to(self.device)
        mesh_grid = F.interpolate(sparse_g.permute(0, 3, 1, 2), size=(dst_h, dst_w), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        
        yy, xx = torch.meshgrid(torch.linspace(-0.5 * (src_h/src_w), 0.5 * (src_h/src_w), src_h, device=self.device), 
                                torch.linspace(-0.5, 0.5, src_w, device=self.device), indexing='ij')
        r2 = xx**2 + yy**2
        dist = 1 + lens.k1 * r2 + lens.k2 * (r2**2)
        lens_grid = torch.stack((xx * 2 * dist, yy * 2 * dist * (src_w/src_h)), dim=-1).unsqueeze(0)
        
        full_grid = F.grid_sample(lens_grid.permute(0, 3, 1, 2), mesh_grid, mode='bilinear', padding_mode='reflection', align_corners=True).permute(0, 2, 3, 1)
        rect_t = F.grid_sample(img_t, full_grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        
        if use_sr:
            with torch.no_grad(): rect_t = self.sr_model(rect_t)
        out_np = rect_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        return np.ascontiguousarray((out_np.clip(0, 1) * 255.0).astype(np.uint8))

    def minimize_energy(self, mesh: MeshModel, fixed_indices, num_iters=50):
        rows, cols = mesh.rows, mesh.cols
        pts_t = torch.tensor([[mesh.points[r][c].x, mesh.points[r][c].y] for r in range(rows) for c in range(cols)], dtype=torch.float32, requires_grad=True, device=self.device)
        idl_t = torch.tensor([[c/(cols-1), r/(rows-1)] for r in range(rows) for c in range(cols)], dtype=torch.float32, device=self.device)
        opt = torch.optim.LBFGS([pts_t], lr=1, max_iter=num_iters)
        def closure():
            opt.zero_grad()
            c_g, i_g = pts_t.view(rows, cols, 2), idl_t.view(rows, cols, 2)
            loss = torch.sum(((c_g[:, 1:]-c_g[:, :-1])-(i_g[:, 1:]-i_g[:, :-1]))**2)+torch.sum(((c_g[1:, :]-c_g[:-1, :])-(i_g[1:, :]-i_g[:-1, :]))**2)
            for r, c in fixed_indices:
                loss += 1000.0 * torch.sum((pts_t[r*cols+c] - torch.tensor([mesh.points[r][c].x, mesh.points[r][c].y], device=self.device))**2)
            loss.backward(); return loss
        opt.step(closure); refined = pts_t.detach().cpu().numpy().reshape(rows, cols, 2)
        for r in range(rows):
            for c in range(cols): mesh.points[r][c].x, mesh.points[r][c].y = float(refined[r,c,0]), float(refined[r,c,1])
