import cv2
import numpy as np
from model import MeshModel, LensModel, Point

class WarpEngine:
    @staticmethod
    def get_mesh_map(mesh: MeshModel, width: int, height: int):
        """
        Generates map_x and map_y for cv2.remap based on the full mesh grid.
        Maps each pixel in the output (rectified) image to its location in the source.
        """
        # grid_pts are the "rectified" coordinates (0,0) to (W,H)
        # mesh.points are the "distorted" coordinates on the source image
        
        # We need to interpolate the mesh points to every pixel in the target image.
        # Target grid points (0...W, 0...H)
        out_rows = mesh.rows
        out_cols = mesh.cols
        
        # Points on the source image (distorted)
        src_points = np.zeros((out_rows, out_cols, 2), dtype=np.float32)
        for r in range(out_rows):
            for c in range(out_cols):
                src_points[r, c] = [mesh.points[r][c].x * (width - 1), 
                                     mesh.points[r][c].y * (height - 1)]
        
        # We want to interpolate these src_points over the entire width x height image.
        # We can use cv2.resize on the small (out_rows x out_cols) grid to the full resolution.
        # This acts as bilinear interpolation.
        
        full_map = cv2.resize(src_points, (width, height), interpolation=cv2.INTER_LINEAR)
        map_x = full_map[:, :, 0].astype(np.float32)
        map_y = full_map[:, :, 1].astype(np.float32)
        
        return map_x, map_y

    @staticmethod
    def apply_lens_distortion(image, lens: LensModel):
        if lens.k1 == 0 and lens.k2 == 0:
            return image
            
        h, w = image.shape[:2]
        distCoeffs = np.array([lens.k1, lens.k2, 0, 0], dtype=np.float32)
        f = max(w, h)
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
        
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w, h), 0)
        mapx, mapy = cv2.initUndistortRectifyMap(K, distCoeffs, None, new_K, (w, h), 5)
        return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    @staticmethod
    def apply_mesh_rectification(image, mesh: MeshModel):
        """Perform full grid-based rectification."""
        h, w = image.shape[:2]
        # In a real tool, we might want to keep the original resolution or some aspect ratio.
        # For preview, we use the image's original dimensions as the target size.
        map_x, map_y = WarpEngine.get_mesh_map(mesh, w, h)
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    @staticmethod
    def sync_perspective(mesh: MeshModel):
        """Update internal points based on the 4 corners using a homography."""
        # Corners: TL, TR, BL, BR
        src_corners = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1]
        ], dtype=np.float32)
        
        dst_corners = np.array([
            [mesh.points[0][0].x, mesh.points[0][0].y],
            [mesh.points[0][-1].x, mesh.points[0][-1].y],
            [mesh.points[-1][0].x, mesh.points[-1][0].y],
            [mesh.points[-1][-1].x, mesh.points[-1][-1].y]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        
        for r in range(mesh.rows):
            for c in range(mesh.cols):
                target_x = c / (mesh.cols - 1)
                target_y = r / (mesh.rows - 1)
                
                pt = np.array([[[target_x, target_y]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(pt, M)
                
                mesh.points[r][c].x = float(transformed[0][0][0])
                mesh.points[r][c].y = float(transformed[0][0][1])

    @staticmethod
    def expand_mesh(mesh: MeshModel):
        """Expand the mesh to cover the largest possible area within the image bounds while preserving perspective."""
        # 1. Get current homography mapping: distorted -> rectified (unit square)
        src_pts = np.array([
            [mesh.points[0][0].x, mesh.points[0][0].y],
            [mesh.points[0][-1].x, mesh.points[0][-1].y],
            [mesh.points[-1][0].x, mesh.points[-1][0].y],
            [mesh.points[-1][-1].x, mesh.points[-1][-1].y]
        ], dtype=np.float32)
        
        dst_pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        
    @staticmethod
    def expand_mesh(mesh: MeshModel):
        """Expand the mesh until it hits the image boundaries while preserving its perspective."""
        # 1. Get current homography mapping: distorted -> rectified (unit square [0,1])
        src_pts = np.array([
            [mesh.points[0][0].x, mesh.points[0][0].y],
            [mesh.points[0][-1].x, mesh.points[0][-1].y],
            [mesh.points[-1][0].x, mesh.points[-1][0].y],
            [mesh.points[-1][-1].x, mesh.points[-1][-1].y]
        ], dtype=np.float32)
        
        dst_pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        
        try:
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            H_inv = np.linalg.inv(H)
        except (cv2.error, np.linalg.LinAlgError):
            return 

        # 2. 拡大率 S を二分探索で求める (1.0 = 現状維持, 10.0 = 超広域)
        # 画像端のどれかにぶつかるまで、パースを維持したまま正規化空間上で広げる
        low = 1.0
        high = 10.0 
        best_s = 1.0
        
        for _ in range(15): # 二分探索 15回で十分な精度
            mid = (low + high) / 2.0
            
            # 正規化空間(0..1)の中心(0.5, 0.5)を基準に mid 倍に広げた四隅を想定
            # new_u = 0.5 + (u - 0.5) * mid
            u0, v0 = 0.5 - 0.5 * mid, 0.5 - 0.5 * mid
            u1, v1 = 0.5 + 0.5 * mid, 0.5 + 0.5 * mid
            
            test_rect = np.array([
                [[u0, v0]], [[u1, v0]], [[u0, v1]], [[u1, v1]]
            ], dtype=np.float32)
            
            # 画像空間（歪みのある空間）へ逆投影
            proj = cv2.perspectiveTransform(test_rect, H_inv)
            
            # すべての点が 0.0 ~ 1.0 に収まっているかチェック
            in_bound = True
            for i in range(4):
                px, py = proj[i][0]
                if px < 0 or px > 1 or py < 0 or py > 1:
                    in_bound = False
                    break
            
            if in_bound:
                best_s = mid
                low = mid
            else:
                high = mid

        # 3. 確定した best_s で四隅を更新
        u0, v0 = 0.5 - 0.5 * best_s, 0.5 - 0.5 * best_s
        u1, v1 = 0.5 + 0.5 * best_s, 0.5 + 0.5 * best_s
        final_rect = np.array([[[u0, v0]], [[u1, v0]], [[u0, v1]], [[u1, v1]]], dtype=np.float32)
        final_proj = cv2.perspectiveTransform(final_rect, H_inv)
        
        mesh.points[0][0].x, mesh.points[0][0].y = final_proj[0][0]
        mesh.points[0][-1].x, mesh.points[0][-1].y = final_proj[1][0]
        mesh.points[-1][0].x, mesh.points[-1][0].y = final_proj[2][0]
        mesh.points[-1][-1].x, mesh.points[-1][-1].y = final_proj[3][0]
        
        # 4. 内部格子を再生成
        WarpEngine.sync_perspective(mesh)

    @staticmethod
    def process_preview(image, mesh: MeshModel, lens: LensModel, rectified=False):
        # 1. Apply Lens correction first (or should it be mesh first? usually lens correction first)
        # Actually, the user aligns the mesh ON the distorted image. 
        # So we should apply lens distortion to the image BEFORE mesh rectification if we want to "correct" both.
        # However, the preview shows the "Rectified" result.
        
        # Step A: Apply lens distortion (undistort)
        temp = WarpEngine.apply_lens_distortion(image, lens)
        
        if rectified:
            # Step B: Apply mesh rectification on the (already lens-corrected) image
            return WarpEngine.apply_mesh_rectification(temp, mesh)
            
        return temp
