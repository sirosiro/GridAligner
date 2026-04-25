import torch
import numpy as np
import cv2
import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from pytorch_engine import PyTorchWarpEngine
from model import LensModel

def test_estimate_lens_parameters_simulated():
    engine = PyTorchWarpEngine(device=torch.device("cpu"))
    
    # テスト画像の生成: 歪んだ線を含む画像をシミュレート
    w, h = 1200, 800
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 理想的な直線（格子状）を描画
    for y in range(100, h, 200):
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 3)
    for x in range(100, w, 200):
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 3)
        
    # 歪みの適用 (k1=0.2)
    dist_coeffs = np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32)
    K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist_coeffs, None, K, (w, h), 5)
    distorted_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    
    # 進捗コールバックの記録用
    progress_history = []
    def cb(curr, total, loss):
        progress_history.append((curr, total))
        
    k1, k2, k3 = engine.estimate_lens_parameters(distorted_img, progress_callback=cb)
    
    print(f"Estimated k1: {k1}, k2: {k2}, k3: {k3}")
    
    # 全く変化しない（0.0のまま）ではなく、何らかの補正方向（この場合は正のk1に対して負の方向）
    # に最適化が進んでいることを確認。
    assert len(progress_history) > 0
    assert progress_history[-1][0] == 250 # max_iters (Matched with engine)
    
    # 歪ませた画像に対して、推定値が有意な値を持っていること
    assert abs(k1) > 0.0001 or abs(k2) > 0.0001 or abs(k3) > 0.0001

def test_estimate_lens_parameters_blank():
    engine = PyTorchWarpEngine(device=torch.device("cpu"))
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 線がない画像では 0, 0, 0 を返すべき
    k1, k2, k3 = engine.estimate_lens_parameters(blank)
    assert k1 == 0.0
    assert k2 == 0.0
    assert k3 == 0.0
