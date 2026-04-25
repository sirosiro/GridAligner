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
        # 残差接続（Skip Connection）の導入: 入力をバイリニアで拡大したものをベースにする
        # これにより、未学習のモデルでも画像が真っ暗になるのを防ぐ
        base = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        # AI による詳細復元パス
        res = torch.tanh(self.conv1(x))
        res = torch.tanh(self.conv2(res))
        res = self.pixel_shuffle(self.conv3(res))
        
        # ベース画像に AI の補正分を微量加算（学習前なので寄与度を抑える）
        return base + res * 0.1

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
        dist = 1 + lens.k1 * r2 + lens.k2 * (r2**2) + lens.k3 * (r2**3)
        return torch.stack((xx * 2 * dist, yy * 2 * dist * (w/h)), dim=-1).unsqueeze(0)

    def reproject_mesh(self, mesh: MeshModel, old_lens: LensModel, new_lens: LensModel, aspect_ratio: float = 1.0):
        """
        レンズパラメータの変更に合わせて、メッシュ制御点の座標を再投影し、
        画像内の物理的な位置（ピクセル）を維持します。
        aspect_ratio = width / height
        """
        rows, cols = mesh.rows, mesh.cols
        pts = []
        for r in range(rows):
            for c in range(cols):
                p = mesh.points[r][c]
                # 1. 現在の正規化座標 -> 中心相対座標 (xx, yy)
                # Engine (process_all) と 100% 一致する座標系
                xx = p.x - 0.5
                yy = (p.y - 0.5) / aspect_ratio
                
                # 2. 旧レンズパラメータで「歪み画像内」の物理的な座標 (ux, uy) を計算
                r2_old = xx**2 + yy**2
                dist_old = 1 + old_lens.k1 * r2_old + old_lens.k2 * (r2_old**2) + old_lens.k3 * (r2_old**3)
                ux, uy = xx * dist_old, yy * dist_old
                
                # 3. 新レンズパラメータにおいて (ux, uy) に対応する「新正規化座標 (nx, ny)」を逆算
                # ニュートン法で近似解を求める
                nx_val, ny_val = xx, yy # 初期値として現在の位置を使用
                for _ in range(10): # 反復回数を増やして精度向上
                    r2_new = nx_val**2 + ny_val**2
                    dist_new = 1 + new_lens.k1 * r2_new + new_lens.k2 * (r2_new**2) + new_lens.k3 * (r2_new**3)
                    
                    # 誤差と更新 (簡易ニュートン)
                    nx_val -= (nx_val * dist_new - ux) / dist_new
                    ny_val -= (ny_val * dist_new - uy) / dist_new
                
                # 4. 新正規化座標 -> 0.0-1.0 空間
                p.x = float(np.clip(nx_val + 0.5, 0.0, 1.0))
                p.y = float(np.clip(ny_val * aspect_ratio + 0.5, 0.0, 1.0))

    def _find_grid_lines(self, hist, size, expected=None):
        sigma = 21 if expected and expected > 10 else 51
        blurred = cv2.GaussianBlur(hist.astype(np.float32), (sigma, 1), 0).flatten()
        peaks = []
        min_gap = size / (expected + 2) if expected else size * 0.03
        thresh = np.mean(blurred) * 1.1
        for i in range(1, len(blurred)-1):
            if blurred[i] > blurred[i-1] and blurred[i] > blurred[i+1] and blurred[i] > thresh:
                if not peaks or abs(i - peaks[-1]) > min_gap:
                    peaks.append(i)
        if not peaks:
            return [size * (i/(expected-1)) for i in range(expected)] if expected else [size*0.1, size*0.9]
        if expected and len(peaks) != expected:
            gap = np.median(np.diff(peaks)) if len(peaks) > 1 else size * 0.1
            while len(peaks) < expected and peaks[0] > gap * 1.5:
                peaks.insert(0, peaks[0] - gap)
            while len(peaks) < expected and peaks[-1] < size - gap * 1.5:
                peaks.append(peaks[-1] + gap)
        return peaks

    def estimate_grid_dimensions(self, image_np):
        h, w = image_np.shape[:2]
        edges = np.zeros((h, w), dtype=np.uint8)
        for i in range(3):
            edges = cv2.bitwise_or(edges, cv2.Canny(image_np[:, :, i], 50, 150))
        y_lines = self._find_grid_lines(np.sum(edges, axis=1), h, expected=None)
        x_lines = self._find_grid_lines(np.sum(edges, axis=0), w, expected=None)
        return len(y_lines), len(x_lines)

    def _compute_linearity_loss(self, segments, k1, k2, k3):
        total_loss = 0.0
        total_tr = 0.0
        for pts in segments:
            xx, yy = pts[:, 0], pts[:, 1]
            r2 = xx**2 + yy**2
            dist = 1 + k1 * r2 + k2 * (r2**2) + k3 * (r2**3)
            ux, uy = xx * dist, yy * dist
            
            points = torch.stack([ux, uy], dim=1)
            mean = points.mean(dim=0)
            diff = points - mean
            cov = torch.matmul(diff.t(), diff) / (len(points) - 1)
            
            tr = cov[0,0] + cov[1,1] + 1e-8
            det = cov[0,0]*cov[1,1] - cov[0,1]*cov[1,0]
            delta = torch.sqrt(torch.clamp(tr**2 - 4*det, min=1e-12))
            l_min = 0.5 * (tr - delta)
            
            total_loss += (l_min / tr)
            total_tr += tr
        return total_loss / len(segments), total_tr / len(segments)

    def estimate_lens_parameters(self, image_np, progress_callback=None):
        """
        堅実な外郭ベースの推定に戻し、暴走を防ぐための強い制約を導入。
        """
        h, w = image_np.shape[:2]
        scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
        small = cv2.resize(image_np, (0,0), fx=scale, fy=scale)
        sh, sw = small.shape[:2]
        
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # 【重要】再び RETR_EXTERNAL に戻し、大きな枠の直線性に集中する
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        segments = []
        for cnt in contours:
            if len(cnt) < 150: continue
            
            # 輪郭を辺ごとに分割
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if 3 <= len(approx) <= 8:
                corner_indices = []
                for p in approx:
                    dists = np.sum((cnt.reshape(-1, 2) - p[0])**2, axis=1)
                    corner_indices.append(np.argmin(dists))
                corner_indices.sort()
                
                for j in range(len(corner_indices)):
                    start = corner_indices[j]
                    end = corner_indices[(j+1)%len(corner_indices)]
                    sub_cnt = np.concatenate([cnt[start:], cnt[:end]], axis=0) if end < start else cnt[start:end]
                    if len(sub_cnt) < 30: continue
                    indices = np.linspace(0, len(sub_cnt)-1, 40).astype(int)
                    pts = [[sub_cnt[idx][0][0]/sw - 0.5, sub_cnt[idx][0][1]/sw - 0.5*(sh/sw)] for idx in indices]
                    segments.append(torch.tensor(pts, device=self.device, dtype=torch.float32))
        
        if not segments:
            return 0.0, 0.0, 0.0

        with torch.no_grad():
            initial_linearity, initial_tr = self._compute_linearity_loss(segments, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
            loss_neg, _ = self._compute_linearity_loss(segments, torch.tensor(-0.1), torch.tensor(0.0), torch.tensor(0.0))
            loss_pos, _ = self._compute_linearity_loss(segments, torch.tensor(0.1), torch.tensor(0.0), torch.tensor(0.0))
            start_k1 = -0.05 if loss_neg < loss_pos else 0.05

        k1 = torch.tensor(start_k1, device=self.device, requires_grad=True)
        k2 = torch.tensor(0.0, device=self.device, requires_grad=True)
        k3 = torch.tensor(0.0, device=self.device, requires_grad=True)
        # 慎重な学習率 (0.01)
        optimizer = torch.optim.Adam([k1, k2, k3], lr=0.01)
        
        max_iters = 250
        for i in range(max_iters):
            optimizer.zero_grad()
            linearity, current_tr = self._compute_linearity_loss(segments, k1, k2, k3)
            
            # 【究極のブレーキ設定】
            # 面積保存ペナルティを 5000.0 に増強。極端な拡大・縮小を「絶対悪」とする。
            total_loss = linearity * 2000.0 + torch.abs(current_tr / initial_tr - 1.0) * 5000.0
            
            # k1, k2, k3 への正則化も強化
            total_loss += k1**2 * 0.1 + k2**2 * 20.0 + k3**2 * 50.0
            
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                # 安全な範囲 (-0.8〜0.8) に制限
                k1.clamp_(-0.8, 0.8)
                k2.clamp_(-0.3, 0.3)
                k3.clamp_(-0.1, 0.1)
            
            if progress_callback and i % 5 == 0:
                progress_callback(i + 1, max_iters, float(total_loss.detach().cpu()))
        
        if progress_callback:
            progress_callback(max_iters, max_iters, float(total_loss.detach().cpu()))
                
        return float(k1.detach().cpu().numpy()), float(k2.detach().cpu().numpy()), float(k3.detach().cpu().numpy())

    def find_outer_corners(self, image_np, target_rows=None, target_cols=None):
        """チェッカーボードを検出し、その四隅の座標を返します。"""
        if image_np is None: return None
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        # 試行するパターンのリスト。targetが指定されていればそれを最優先する。
        # OpenCVのチェッカーボード検出は、内部の交点数（セルの数 - 1）を指定する必要がある。
        patterns = []
        if target_rows and target_cols:
            patterns.append((target_cols - 1, target_rows - 1))
        
        # 一般的なチェッカーボードサイズもフォールバックとして保持
        patterns.extend([(9, 7), (8, 6), (11, 8), (7, 5), (5, 5)])
        
        for p_cols, p_rows in patterns:
            if p_cols <= 1 or p_rows <= 1: continue
            ret, corners = cv2.findChessboardCorners(gray, (p_cols, p_rows), 
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
            if ret:
                # サブピクセル精度で補正
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                
                # 検出された格子の四隅を抽出
                corners = corners.reshape(p_rows, p_cols, 2)
                tl = corners[0, 0]
                tr = corners[0, -1]
                br = corners[-1, -1]
                bl = corners[-1, 0]
                return [tl, tr, br, bl]
        return None

    def detect_initial_grid(self, image_np, target_rows=None, target_cols=None):
        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
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

    def apply_mesh_rectification(self, image_np, mesh: MeshModel):
        return self.process_all(image_np, mesh, LensModel(), use_sr=False)

    def process_all(self, image_np, mesh: MeshModel, lens: LensModel, use_sr: bool):
        src_h, src_w = image_np.shape[:2]
        dst_w, dst_h = self.get_target_size(src_w, src_h, mesh)
        img_t = torch.from_numpy(np.ascontiguousarray(image_np.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device)/255.0
        rows, cols = mesh.rows, mesh.cols
        sparse_g = torch.stack([torch.tensor([mesh.points[r][c].x*2-1, mesh.points[r][c].y*2-1]) for r in range(rows) for c in range(cols)]).view(1, rows, cols, 2).to(self.device)
        mesh_grid = F.interpolate(sparse_g.permute(0, 3, 1, 2), size=(dst_h, dst_w), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        yy, xx = torch.meshgrid(torch.linspace(-0.5 * (src_h/src_w), 0.5 * (src_h/src_w), src_h, device=self.device), torch.linspace(-0.5, 0.5, src_w, device=self.device), indexing='ij')
        r2 = xx**2 + yy**2
        dist = 1 + lens.k1 * r2 + lens.k2 * (r2**2) + lens.k3 * (r2**3)
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
