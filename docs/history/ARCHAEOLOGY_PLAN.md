# ARCHAEOLOGY_PLAN.md

## 1. プロジェクトの目的とドメインの特定 (Domain Identification)

- **ドメイン**: 画像幾何学歪み補正 (Geometric Image Distortion Correction)
- **概要**: 魚眼・広角レンズの歪みおよびパース歪みを、GUI（メッシュ操作）と AI（格子検出）を組み合わせて補正するデスクトップアプリケーション。
- **主要な外部規約**: 
    - OpenCV 幾何変換規格 (Homography, Radial Distortion Model)
    - PyTorch Tensor 演算規格 (`grid_sample`)
    - PySide6 (Qt) イベントループ・シグナルスロット規約

## 2. マニフェスト配置マップ (Manifest Map)

- **[Root] `./ARCHITECTURE_MANIFEST.md`**: 
    - プロジェクト全体の不変の原則、AI 協調指針、高レベルなデータフロー（正規化座標系）を定義。
    - サブマニフェストへのポインタを配置。
- **[Src] `./src/ARCHITECTURE_MANIFEST.md`**: 
    - MVC 各レイヤー（Model, View, Controller）および Engine の詳細な責務、API シグネチャ、ライフサイクル管理を定義。

## 3. 各階層の推定責務 (Estimated Responsibilities)

- **Root (./)**: 
    - 「座標の正規化（0.0-1.0）」という絶対原則の維持。
    - レンズ補正とメッシュ補正のパイプライン順序の保証。
- **Src (./src)**:
    - `Model`: ステートレスなデータ保持。
    - `View`: UI イベントの正規化と Signal 発行。
    - `Controller`: レイヤー間の同期とライフサイクル制御。
    - `Engine`: 画像処理ロジックのステートレスな実装。

---

## Source Analysis Metadata

- **Source Repository**: GridAligner
- **Detected License**: MIT (LICENSE ファイルより)
- **Structural Similarity Risk**: Low (標準的な MVC パターン)
- **Attribution Required**: No (内部プロジェクト)

---
上記の発掘計画に従い、既存プロジェクトの解析を進めてもよろしいでしょうか？
承認をいただければ、各ソースコードの詳細解析とマニフェストの再構築（Step 5）を開始します。
