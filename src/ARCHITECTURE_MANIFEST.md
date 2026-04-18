# src/ARCHITECTURE_MANIFEST.md (Implementation Detail)

## Part 1: このマニフェストの取扱説明書 (Guide)

1.  **目的 (Purpose)**: 
    - `src/` 以下のソースコード（Python/PySide6）における、具体的な API 設計、データ構造、およびコンポーネント間の契約（Contract）を定義する。
2.  **憲章の書き方 (Guidelines)**:
    - **API シグネチャの明記**: 引数の意味、型、および所有権（Ownership）を記述する。
    - **ライフサイクルの定義**: オブジェクトがいつ生成され、いつ破棄されるかを明確にする。

---

## Part 2: マニフェスト本体 (Content)

### 1. 核となる原則 (Core Principles)

- **ステートレスな Engine**: Engine 階層の関数は内部状態を持たず、Model を外部から受け取って処理結果を返す（Side-effect free）設計を維持する。
- **UI イベントの正規化**: View (Canvas) から Controller へ送られる座標は、常にスクリーン座標ではなく画像相対の正規化座標 (0.0-1.0) である。

### 4. コンポーネント設計仕様 (Component Design Specifications)

#### 4.1. Model レイヤー (`model.py`)
- **責務**: アプリケーションの幾何学的状態（Mesh, Lens）の保持と、JSON 形式へのシリアライズ。
- **主要クラス**:
    - `MeshModel`: $N \times M$ の `Point` 2次元配列を管理。
        - `rotate_clockwise()`: 格子構造を転置・反転させ、物理的な「上」を補正する。
- **データ所有権**: Model は Controller によって管理され、View や Engine へは参照として渡される。

#### 4.2. Engine レイヤー (`engine.py`, `pytorch_engine.py`)
- **責務**: 数学的な幾何変換ロジックの実装。
- **主要な API**:
    - `WarpEngine.get_mesh_map(mesh: MeshModel, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]`:
        - 低解像度グリッドからフル解像度の座標マップ（map_x, map_y）を生成する。
    - `PyTorchWarpEngine.minimize_energy(mesh: MeshModel, fixed_indices: List[Tuple[int, int]])`:
        - L-BFGS を用いて格子の曲げエネルギーを最小化し、滑らかな変形を実現する。
- **技術的制約**: OpenCV と PyTorch の間で、中心座標の定義や補完のアルゴリズムが 100% 一致することを保証しなければならない。

#### 4.3. Controller レイヤー (`controller.py`)
- **責務**: レイヤー間のオーケストレーション。View からの Signal を受け取り、Model を更新し、Engine を通じてプレビューを再生成する。
- **同期ポリシー**: `on_point_moved` 時、四隅が移動した場合は `WarpEngine.sync_perspective` を強制実行し、射影変換の整合性を維持する。

#### 4.4. View レイヤー (`view.py`)
- **責務**: 画像と格子の重畳描画、およびマウス操作による座標入力の取得。
- **主要な API**:
    - `Canvas.pointMoved(int, int, float, float)`: 移動した制御点の (row, col) と、新しい正規化座標 (x, y) を発行する。
- **視認性補助**: `show_grid`, `show_crosshair` フラグにより、レンズ補正とメッシュ補正のそれぞれのフェーズに適したガイドラインを表示する。

### 5. 既知の未解決課題 (Known Open Issues)

<!-- Issue: PyTorch デバイス（CPU/GPU）の動的な切り替えUI, Status: 保留, Rationale: 現在は自動選択。将来的にユーザーが選択できるように拡張予定。 -->
