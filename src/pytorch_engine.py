import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from model import MeshModel, LensModel, Point

# Mac環境でのOpenCLキャッシュ警告を抑制
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
            
    def apply_mesh_rectification(self, image_np, mesh: MeshModel):
        h, w = image_np.shape[:2]
        img_tensor = torch.from_numpy(np.ascontiguousarray(image_np.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device)
        rows, cols = mesh.rows, mesh.cols
        sparse_grid = torch.stack([torch.tensor([mesh.points[r][c].x * 2 - 1, mesh.points[r][c].y * 2 - 1]) 
                                  for r in range(rows) for c in range(cols)]).view(1, rows, cols, 2).to(self.device)
        full_grid = F.interpolate(sparse_grid.permute(0, 3, 1, 2), size=(h, w), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        output = F.grid_sample(img_tensor, full_grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        return np.ascontiguousarray(output.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8))

    def detect_initial_grid(self, image_np, target_rows=None, target_cols=None):
        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        # --- Pattern A: Checkerboard Detection (Flexible size matching) ---
        if target_rows and target_cols:
            candidates = [
                (target_cols - 2, target_rows - 2),
                (target_cols - 1, target_rows - 1),
                (target_cols, target_rows)
            ]
            
            for p_cols, p_rows in candidates:
                # 認識アルゴリズムの最小制約
                if p_cols <= 1 or p_rows <= 1: continue
                pattern_size = (p_cols, p_rows)
                
                try:
                    ret = False
                    # findChessboardCornersSB は 3x3 以上の交点が必要
                    if p_cols > 2 and p_rows > 2 and hasattr(cv2, 'findChessboardCornersSB'):
                        ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, 
                            cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
                    
                    if not ret:
                        ret, corners = cv2.findChessboardCorners(gray, pattern_size, 
                            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
                    
                    if ret:
                        if not hasattr(cv2, 'findChessboardCornersSB') or len(corners.shape) == 3:
                            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        
                        dst_pts = corners.reshape(-1, 2)
                        src_pts = []
                        for r in range(p_rows):
                            for c in range(p_cols): src_pts.append([c, r])
                        src_pts = np.array(src_pts, dtype=np.float32)
                        
                        M, _ = cv2.findHomography(src_pts, dst_pts)
                        if M is not None:
                            res_pts = []
                            offset_r = (target_rows - p_rows) / 2.0
                            offset_c = (target_cols - p_cols) / 2.0
                            for r in range(target_rows):
                                row = []
                                for c in range(target_cols):
                                    grid_pt = np.array([[[c - offset_c, r - offset_r]]], dtype=np.float32)
                                    transformed = cv2.perspectiveTransform(grid_pt, M)[0][0]
                                    px, py = transformed
                                    row.append(Point(float(max(0, min(w, px))/w), float(max(0, min(h, py))/h)))
                                res_pts.append(row)
                            return res_pts, target_rows, target_cols
                except Exception:
                    # 認識中のエラーは無視して次の候補または Fallback へ
                    continue

        # --- Pattern B: Statistical Profile Analysis (Fallback) ---
        edges_sum = np.zeros((h, w), dtype=np.uint8)
        for i in range(3):
            edges_sum = cv2.bitwise_or(edges_sum, cv2.Canny(image_np[:, :, i], 50, 150))
        
        def find_grid_lines(hist, size, expected_cnt=None):
            sigma = 21 if expected_cnt and expected_cnt > 10 else 51
            blurred = cv2.GaussianBlur(hist.astype(np.float32), (sigma, 1), 0).flatten()
            peaks = []
            min_gap = size / (expected_cnt + 2) if expected_cnt else size * 0.03
            threshold = np.mean(blurred) * 1.1
            for i in range(1, len(blurred) - 1):
                if blurred[i] > blurred[i-1] and blurred[i] > blurred[i+1]:
                    if blurred[i] > threshold:
                        if not peaks or abs(i - peaks[-1]) > min_gap:
                            peaks.append(i)
            
            if not peaks: return [size * (i/(target_rows-1)) for i in range(target_rows)] if target_rows else [size*0.1, size*0.9]
            gap = np.median(np.diff(peaks)) if len(peaks) > 1 else size * 0.1
            while peaks[0] > gap * 1.5: peaks.insert(0, peaks[0] - gap)
            while peaks[-1] < size - gap * 1.5: peaks.append(peaks[-1] + gap)
            return peaks

        y_lines = find_grid_lines(np.sum(edges_sum, axis=1), h, target_rows)
        x_lines = find_grid_lines(np.sum(edges_sum, axis=0), w, target_cols)
        rows, cols = len(y_lines), len(x_lines)
        
        crude_pts = []
        for y in y_lines:
            for x in x_lines: crude_pts.append([x, y])
        crude_pts = np.array(crude_pts, dtype=np.float32).reshape(-1, 1, 2)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        try:
            refined_pts = cv2.cornerSubPix(edges_sum, crude_pts, (5, 5), (-1, -1), criteria)
            refined_pts = refined_pts.reshape(rows, cols, 2)
        except:
            refined_pts = crude_pts.reshape(rows, cols, 2)
        
        res_pts = []
        for r in range(rows):
            row = []
            for c in range(cols):
                px, py = refined_pts[r, c]
                row.append(Point(float(max(0, min(w, px))/w), float(max(0, min(h, py))/h)))
            res_pts.append(row)
            
        return res_pts, rows, cols

    def apply_lens_distortion(self, img_tensor, lens: LensModel):
        if lens.k1 == 0 and lens.k2 == 0: return img_tensor
        _, _, h, w = img_tensor.shape
        yy, xx = torch.meshgrid(torch.linspace(-0.5 * (h/w), 0.5 * (h/w), h, device=self.device), 
                                torch.linspace(-0.5, 0.5, w, device=self.device), indexing='ij')
        r2 = xx**2 + yy**2
        dist = 1 + lens.k1 * r2 + lens.k2 * (r2**2)
        grid = torch.stack((xx * 2 * dist, yy * 2 * dist * (w/h)), dim=-1).unsqueeze(0)
        return F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    def minimize_energy(self, mesh: MeshModel, fixed_indices, num_iters=50):
        rows, cols = mesh.rows, mesh.cols
        pts = [[mesh.points[r][c].x, mesh.points[r][c].y] for r in range(rows) for c in range(cols)]
        pts_t = torch.tensor(pts, dtype=torch.float32, requires_grad=True, device=self.device)
        idl_t = torch.tensor([[c/(cols-1), r/(rows-1)] for r in range(rows) for c in range(cols)], dtype=torch.float32, device=self.device)
        opt = torch.optim.LBFGS([pts_t], lr=1, max_iter=num_iters)
        def closure():
            opt.zero_grad()
            c_g, i_g = pts_t.view(rows, cols, 2), idl_t.view(rows, cols, 2)
            loss = torch.sum(((c_g[:, 1:]-c_g[:, :-1])-(i_g[:, 1:]-i_g[:, :-1]))**2)+torch.sum(((c_g[1:, :]-c_g[:-1, :])-(i_g[1:, :]-i_g[:-1, :]))**2)
            for r, c in fixed_indices:
                idx = r * cols + c
                tgt = torch.tensor([mesh.points[r][c].x, mesh.points[r][c].y], device=self.device)
                loss += 1000.0 * torch.sum((pts_t[idx] - tgt)**2)
            loss.backward(); return loss
        opt.step(closure); refined = pts_t.detach().cpu().numpy().reshape(rows, cols, 2)
        for r in range(rows):
            for c in range(cols): mesh.points[r][c].x, mesh.points[r][c].y = float(refined[r,c,0]), float(refined[r,c,1])

    def process_all(self, image_np, mesh: MeshModel, lens: LensModel, use_sr: bool):
        h, w = image_np.shape[:2]
        img_t = torch.from_numpy(np.ascontiguousarray(image_np.transpose(2, 0, 1))).float().unsqueeze(0).to(self.device)/255.0
        img_t = self.apply_lens_distortion(img_t, lens)
        rows, cols = mesh.rows, mesh.cols
        sparse_g = torch.stack([torch.tensor([mesh.points[r][c].x*2-1, mesh.points[r][c].y*2-1]) for r in range(rows) for c in range(cols)]).view(1, rows, cols, 2).to(self.device)
        full_g = F.interpolate(sparse_g.permute(0, 3, 1, 2), size=(h, w), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        rect_t = F.grid_sample(img_t, full_g, mode='bilinear', padding_mode='reflection', align_corners=True)
        if use_sr:
            with torch.no_grad(): rect_t = self.sr_model(rect_t)
        out_np = rect_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        return np.ascontiguousarray((out_np.clip(0, 1) * 255.0).astype(np.uint8))
