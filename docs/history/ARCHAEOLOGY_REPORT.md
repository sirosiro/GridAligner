# ARCHAEOLOGY_REPORT.md

## 0. ドメインの特定と外部規約の宣言
- **ドメイン**: 画像幾何学歪み補正 (Image Rectification)
- **準拠規約**:
    - OpenCV 4.x Camera Calibration & 2D Reconstruction
    - PyTorch Grid Sample (Bilinear Interpolation)
    - Qt (PySide6) Signal/Slot Architecture

## 1. 推定される設計原則 (Core Principles)
- **絶対正規化座標系**: 全ての制御点は 0.0 ~ 1.0 の浮動小数点で管理。`Canvas` 描画時および `Engine` 演算時の解像度依存を完全に排除している。
- **ステートレス・エンジン**: `WarpEngine` および `PyTorchWarpEngine` は状態を持たず、Model を受け取って処理結果を返す純粋関数的な振る舞いをしている。
- **レンズ補正先行フロー**: 放射歪み（Lens）を先に除去してから射影歪み（Mesh/AI）を処理することで、AI の直線認識精度を最大化している。

## 2. 発見されたアーキテクチャ制約
- **UI 抽象化の徹底**: `MainWindow` がウィジェットを直接公開せず、`get_lens_params` 等のゲッターを介して Controller と通信している。これは将来的な UI フレームワーク変更への耐性を示唆。
- **エンジンの冗長性と同期**: OpenCV と PyTorch で同等の計算（k1, k2 歪み）を実装。メイン画面の軽快さとプレビューの高品質化を両立させている。

## 3. 発見された例外と矛盾
- `WarpEngine.expand_mesh` の一部のロジックが `src/engine.py` 内で二重定義（重複）している箇所がある。
- `model.py` 内の `rotate_clockwise` コメントに「FFmpeg 出力時の反転防止」という具体的なユースケースが記述されており、単なる画像変換ツール以上の「プリプロセッサ」としての意図が読み取れる。

## 4. 利用されていないコード (Dead Code) の可能性
- `view.py` 内の `Slot` デコレータの一部が、シグナル接続（`connect`）で直接呼び出されているため、明示的な Slot 定義が冗長になっている可能性がある。

## 5. コンポーネント設計仕様の具体化

### 5.1. Model 層
- `MeshModel`: $N \times M$ の `Point` 配列を保持。`rotate_clockwise` による座標系の転置・反転をサポート。
- `LensModel`: ブラウン・コンラディ（Brown-Conrady）モデルの $k_1, k_2$ 係数を保持。

### 5.2. Engine 層
- `WarpEngine (Static)`: `cv2.getPerspectiveTransform` を用いたホモグラフィ同期と、`cv2.resize` を利用したメッシュの双線形補間（Map 生成）を実装。
- `PyTorchWarpEngine`: L-BFGS 最適化によるメッシュエネルギー最小化 (`minimize_energy`)。チェッカーボード認識とプロファイル解析のハイブリッド検出。

### 5.3. Controller 層
- 全てのライフサイクルと UI 同期を統括。`on_point_moved` でのリアルタイム同期が核。

---

## Source Analysis Metadata

- **Source Repository**: GridAligner
- **Detected License**: MIT
- **Structural Similarity Risk**: Low
- **Attribution Required**: No
