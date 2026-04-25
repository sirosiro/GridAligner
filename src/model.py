import json
import cv2
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

@dataclass
class Point:
    x: float  # Normalized 0.0 - 1.0
    y: float

@dataclass
class LensModel:
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        # 過去の保存データ（k3なし）との互換性を維持
        return cls(
            k1=data.get("k1", 0.0),
            k2=data.get("k2", 0.0),
            k3=data.get("k3", 0.0)
        )

@dataclass
class MeshModel:
    rows: int
    cols: int
    points: List[List[Point]] = field(default_factory=list)

    def __post_init__(self):
        if not self.points:
            self.reset()

    def reset(self):
        self.points = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                row.append(Point(c / (self.cols - 1), r / (self.rows - 1)))
            self.points.append(row)

    def rotate_clockwise(self):
        """
        格子全体を時計回りに90度回転させます。
        AI（OpenCV等）が上下逆さまや横向きに格子を検出してしまった際、
        FFmpeg等の外部出力で映像が反転するのを防ぐために必要です。
        """
        new_rows = self.cols
        new_cols = self.rows
        
        # 新しい配列を確保して転置と反転を行う
        temp_points = [[None for _ in range(new_cols)] for _ in range(new_rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                # 回転行列の計算に相当
                temp_points[c][(self.rows - 1) - r] = self.points[r][c]
        
        self.points = temp_points
        self.rows = new_rows
        self.cols = new_cols

    def subdivide(self):
        """現在のメッシュの解像度を倍（Rows*2-1, Cols*2-1）に細分化します。"""
        new_rows = self.rows * 2 - 1
        new_cols = self.cols * 2 - 1
        new_points = []
        
        for r in range(new_rows):
            row = []
            for c in range(new_cols):
                # 元のインデックスに対応するかどうか
                if r % 2 == 0 and c % 2 == 0:
                    # 既存の点
                    row.append(self.points[r // 2][c // 2])
                elif r % 2 == 0:
                    # 水平方向の中間点
                    p1 = self.points[r // 2][c // 2]
                    p2 = self.points[r // 2][c // 2 + 1]
                    row.append(Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2))
                elif c % 2 == 0:
                    # 垂直方向の中間点
                    p1 = self.points[r // 2][c // 2]
                    p2 = self.points[r // 2 + 1][c // 2]
                    row.append(Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2))
                else:
                    # セルの中心点
                    p1 = self.points[r // 2][c // 2]
                    p2 = self.points[r // 2 + 1][c // 2 + 1]
                    p3 = self.points[r // 2 + 1][c // 2]
                    p4 = self.points[r // 2][c // 2 + 1]
                    row.append(Point((p1.x + p2.x + p3.x + p4.x) / 4, (p1.y + p2.y + p3.y + p4.y) / 4))
            new_points.append(row)
            
        self.rows = new_rows
        self.cols = new_cols
        self.points = new_points

    def transform_by_corners(self, dst_corners: List[Point], linearize: bool = False):
        """
        四隅を指定された dst_corners に合わせるように変形します。
        linearize=True の場合、内部の歪みをリセットして完璧な直線格子にします。
        linearize=False の場合、既存の歪み（曲線）を維持したままパース変形します。
        dst_corners: [TL, TR, BR, BL]
        """
        if len(dst_corners) != 4: return
        
        # ターゲットの四隅
        target = np.array([[p.x, p.y] for p in dst_corners], dtype=np.float32)

        if linearize:
            # 直線化モード: 理想的な 0.0-1.0 グリッドからの変換
            src_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            H, _ = cv2.findHomography(src_corners, target)
            if H is None: return
            
            for r in range(self.rows):
                for c in range(self.cols):
                    tx = c / (self.cols - 1) if self.cols > 1 else 0.5
                    ty = r / (self.rows - 1) if self.rows > 1 else 0.5
                    pts = np.array([[tx, ty]], dtype=np.float32).reshape(-1, 1, 2)
                    transformed = cv2.perspectiveTransform(pts, H)
                    self.points[r][c].x = float(transformed[0][0][0])
                    self.points[r][c].y = float(transformed[0][0][1])
        else:
            # 歪み維持モード: 現在の四隅からの相対変換
            src_corners = np.array([
                [self.points[0][0].x, self.points[0][0].y],
                [self.points[0][self.cols-1].x, self.points[0][self.cols-1].y],
                [self.points[self.rows-1][self.cols-1].x, self.points[self.rows-1][self.cols-1].y],
                [self.points[self.rows-1][0].x, self.points[self.rows-1][0].y]
            ], dtype=np.float32)
            
            H, _ = cv2.findHomography(src_corners, target)
            if H is None: return
            
            for r in range(self.rows):
                for c in range(self.cols):
                    p = self.points[r][c]
                    pts = np.array([[p.x, p.y]], dtype=np.float32).reshape(-1, 1, 2)
                    transformed = cv2.perspectiveTransform(pts, H)
                    p.x = float(transformed[0][0][0])
                    p.y = float(transformed[0][0][1])

    def to_dict(self):
        return {
            "rows": self.rows,
            "cols": self.cols,
            "points": [[asdict(p) for p in row] for row in self.points]
        }

    @classmethod
    def from_dict(cls, data):
        points = [[Point(**p) for p in row] for row in data["points"]]
        return cls(rows=data["rows"], cols=data["cols"], points=points)
