import json
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

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

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
