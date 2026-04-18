import cv2
import numpy as np
from model import MeshModel, LensModel, Point

class WarpEngine:
    @staticmethod
    def get_lens_maps(width, height, lens: LensModel):
        distCoeffs = np.array([lens.k1, lens.k2, 0, 0], dtype=np.float32)
        f = max(width, height)
        K = np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1]], dtype=np.float32)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (width, height), 0)
        map_x, map_y = cv2.initUndistortRectifyMap(K, distCoeffs, None, new_K, (width, height), cv2.CV_32FC1)
        return map_x, map_y

    @staticmethod
    def get_mesh_map(mesh: MeshModel, src_w: int, src_h: int, dst_w: int, dst_h: int, lens: LensModel = None):
        out_rows, out_cols = mesh.rows, mesh.cols
        mesh_pts = np.zeros((out_rows, out_cols, 2), dtype=np.float32)
        for r in range(out_rows):
            for c in range(out_cols):
                mesh_pts[r, c] = [mesh.points[r][c].x * (src_w - 1), 
                                   mesh.points[r][c].y * (src_h - 1)]
        full_mesh_map = cv2.resize(mesh_pts, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        mesh_x = full_mesh_map[:, :, 0]
        mesh_y = full_mesh_map[:, :, 1]
        if lens and (lens.k1 != 0 or lens.k2 != 0):
            lens_map_x, lens_map_y = WarpEngine.get_lens_maps(src_w, src_h, lens)
            final_map_x = cv2.remap(lens_map_x, mesh_x, mesh_y, cv2.INTER_LINEAR)
            final_map_y = cv2.remap(lens_map_y, mesh_x, mesh_y, cv2.INTER_LINEAR)
            return final_map_x, final_map_y
        return mesh_x, mesh_y

    @staticmethod
    def apply_lens_distortion(image, lens: LensModel):
        if lens.k1 == 0 and lens.k2 == 0: return image
        h, w = image.shape[:2]
        map_x, map_y = WarpEngine.get_lens_maps(w, h, lens)
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    @staticmethod
    def apply_mesh_rectification(image, mesh: MeshModel, lens: LensModel = None):
        if image is None: return None
        src_h, src_w = image.shape[:2]
        dst_aspect = (mesh.cols - 1) / (mesh.rows - 1) if mesh.rows > 1 else 1.0
        if dst_aspect > 1.0:
            dst_w, dst_h = src_w, int(src_w / dst_aspect)
        else:
            dst_h, dst_w = src_h, int(src_h * dst_aspect)
        map_x, map_y = WarpEngine.get_mesh_map(mesh, src_w, src_h, dst_w, dst_h, lens)
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    @staticmethod
    def sync_perspective(mesh: MeshModel):
        src = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.float32)
        dst = np.array([[mesh.points[0][0].x, mesh.points[0][0].y],
                        [mesh.points[0][-1].x, mesh.points[0][-1].y],
                        [mesh.points[-1][0].x, mesh.points[-1][0].y],
                        [mesh.points[-1][-1].x, mesh.points[-1][-1].y]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        for r in range(mesh.rows):
            for c in range(mesh.cols):
                tx = c / (mesh.cols - 1) if mesh.cols > 1 else 0.5
                ty = r / (mesh.rows - 1) if mesh.rows > 1 else 0.5
                p = cv2.perspectiveTransform(np.array([[[tx,ty]]], dtype=np.float32), M)
                mesh.points[r][c].x, mesh.points[r][c].y = float(p[0][0][0]), float(p[0][0][1])

    @staticmethod
    def expand_mesh(mesh: MeshModel, img_w: int, img_h: int):
        """
        Safety Directional Expansion:
        Expands in each of 4 directions independently until any point in the next
        row/column would fall outside the FOV. This prevents point piling at the edges.
        """
        # Get current corners in PIXELS
        src_pts = np.array([[mesh.points[0][0].x * img_w, mesh.points[0][0].y * img_h],
                            [mesh.points[0][-1].x * img_w, mesh.points[0][-1].y * img_h],
                            [mesh.points[-1][0].x * img_w, mesh.points[-1][0].y * img_h],
                            [mesh.points[-1][-1].x * img_w, mesh.points[-1][-1].y * img_h]], dtype=np.float32)
        
        # Grid indices corresponding to these points (0 to C-1, 0 to R-1)
        grid_pts = np.array([[0, 0], [mesh.cols - 1, 0], 
                             [0, mesh.rows - 1], [mesh.cols - 1, mesh.rows - 1]], dtype=np.float32)
        
        try:
            # H maps (x,y) pixels to (u,v) grid coordinates
            H = cv2.getPerspectiveTransform(src_pts, grid_pts)
            H_inv = np.linalg.inv(H)
        except: return 

        def is_within_fov(u, v):
            # Transform grid coord (u,v) back to pixel coord
            px = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), H_inv)[0][0]
            return 0 <= px[0] <= img_w and 0 <= px[1] <= img_h

        def check_row_fov(v, u_start, u_end):
            # Check multiple points across the row to be safe
            for u in np.linspace(u_start, u_end, 5):
                if not is_within_fov(u, v): return False
            return True

        def check_col_fov(u, v_start, v_end):
            # Check multiple points across the column to be safe
            for v in np.linspace(v_start, v_end, 5):
                if not is_within_fov(u, v): return False
            return True

        # Initial relative grid bounds
        u_start, u_end = 0.0, float(mesh.cols - 1)
        v_start, v_end = 0.0, float(mesh.rows - 1)

        # Iteratively expand each direction until failure
        changed = True
        while changed:
            changed = False
            # Try LEFT (Decrease u_start)
            if check_col_fov(u_start - 1, v_start, v_end):
                u_start -= 1; changed = True
            # Try RIGHT (Increase u_end)
            if check_col_fov(u_end + 1, v_start, v_end):
                u_end += 1; changed = True
            # Try TOP (Decrease v_start)
            if check_row_fov(v_start - 1, u_start, u_end):
                v_start -= 1; changed = True
            # Try BOTTOM (Increase v_end)
            if check_row_fov(v_end + 1, u_start, u_end):
                v_end += 1; changed = True
            
            # Guard to prevent infinite or extreme growth
            if (u_end - u_start) > 100 or (v_end - v_start) > 100:
                break

        # Calculate final dimensions
        new_cols = int(round(u_end - u_start)) + 1
        new_rows = int(round(v_end - v_start)) + 1
        
        mesh.rows = new_rows
        mesh.cols = new_cols
        mesh.reset()
        
        # Map grid [0..C-1] back to [u_start..u_end] and then to image
        for r in range(new_rows):
            for c in range(new_cols):
                u, v = u_start + c, v_start + r
                px = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), H_inv)[0][0]
                mesh.points[r][c].x = float(max(0, min(img_w, px[0])) / img_w)
                mesh.points[r][c].y = float(max(0, min(img_h, px[1])) / img_h)

    @staticmethod
    def process_preview(image, mesh: MeshModel, lens: LensModel, rectified=False):
        if rectified: return WarpEngine.apply_mesh_rectification(image, mesh, lens)
        return WarpEngine.apply_lens_distortion(image, lens)
