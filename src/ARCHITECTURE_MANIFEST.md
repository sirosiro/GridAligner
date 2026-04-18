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

#### 4.1. クラス構造概略 (Class Diagram)

```mermaid
classDiagram
    namespace Model_Layer {
        class Point {
            +float x
            +float y
        }
        class LensModel {
            +float k1
            +float k2
            +to_dict()
            +from_dict(data)
        }
        class MeshModel {
            +int rows
            +int cols
            +List~List~Point~~ points
            +reset()
            +rotate_clockwise()
            +to_dict()
            +from_dict(data)
        }
    }

    namespace Engine_Layer {
        class WarpEngine {
            <<static>>
            +get_mesh_map(mesh, width, height)
            +apply_lens_distortion(image, lens)
            +apply_mesh_rectification(image, mesh)
            +process_preview(image, mesh, lens, rectified)
            +sync_perspective(mesh)
            +expand_mesh(mesh)
        }
        class PyTorchWarpEngine {
            -device
            -SuperResModel sr_model
            +apply_mesh_rectification(image_np, mesh)
            +detect_initial_grid(image_np, target_rows, target_cols)
            +apply_lens_distortion(img_tensor, lens)
            +minimize_energy(mesh, fixed_indices)
            +process_all(image_np, mesh, lens, use_sr)
        }
        class SuperResModel {
            +forward(x)
        }
    }

    namespace View_Layer {
        class Canvas {
            +QPixmap pixmap
            +MeshModel mesh
            +Point selected_point
            +Point crosshair_pos
            +bool show_grid
            +bool show_crosshair
            +bool constraint_mode
            +Signal pointMoved(int, int, float, float)
            +set_image(qimage)
            +set_mesh(mesh)
            +set_show_grid(bool)
            +set_show_crosshair(bool)
            +draw_grid(painter, pixmap, ox, oy)
            +mousePressEvent(event)
            +mouseMoveEvent(event)
            +mouseReleaseEvent(event)
        }
        class CameraDialog {
            +QLabel video_label
            +QPushButton btn_capture
            +Signal imageCaptured
            +start_camera()
            +stop_camera()
            +update_frame()
            +capture()
        }
        class PreviewWindow {
            +QLabel label
            +set_image(qimage)
        }
        class MainWindow {
            +Canvas canvas
            +get_grid_dimensions()
            +set_grid_dimensions(r, c)
            +get_lens_params()
            +set_lens_params(k1, k2)
            +is_preview_enabled()
            +is_super_res_enabled()
            +is_smooth_warp_enabled()
            +reset_lens_sliders()
            +rotate_grid()
            +set_grid_visible(bool)
            +set_crosshair_visible(bool)
        }
    }

    namespace Controller_Layer {
        class Controller {
            -MainWindow view
            -MeshModel mesh
            -LensModel lens
            -PyTorchWarpEngine py_engine
            -PreviewWindow preview_window
            -CameraDialog camera_dialog
            -Mat original_image
            +open_image()
            +open_camera()
            +load_new_image(image)
            +update_lens()
            +update_preview()
            +resize_mesh()
            +reset_mesh()
            +straighten_grid()
            +expand_mesh_to_full_frame()
            +rotate_grid()
            +on_point_moved(r, c, nx, ny)
            +on_auto_grid()
            +save_project()
            +load_project()
            +export()
            +toggle_preview_window(state)
        }
    }

    %% Relationships
    MeshModel "1" *-- "N" Point : contains
    Controller "1" o-- "1" MainWindow : view
    Controller "1" o-- "1" MeshModel : model
    Controller "1" o-- "1" LensModel : model
    Controller "1" o-- "1" PyTorchWarpEngine : engine
    PyTorchWarpEngine "1" o-- "1" SuperResModel : uses
    Controller ..> WarpEngine : uses
    MainWindow "1" *-- "1" Canvas : contains
    Controller "1" -- "0..1" CameraDialog : creates/manages
    Controller "1" -- "0..1" PreviewWindow : creates/manages
    Canvas ..> MeshModel : displays for drawing
```

#### 4.2. 各クラスの役割詳細 (Component Roles)

##### Model レイヤー (`model.py`)
- **Point**: 2次元座標（x, y）を保持する最小単位のデータクラス。画像サイズに依存しない正規化された中心座標（0.0 - 1.0）として扱う。
- **LensModel**: レンズ歪み補正係数（k1, k2）を保持。プロジェクト保存用のシリアライズ（to_dict/from_dict）を備える。
- **MeshModel**: 格子状メッシュの行数・列数、および全制御点のPointリストを保持。`rotate_clockwise` による座標系の整合性補正を担当。

##### Engine レイヤー (`engine.py`, `pytorch_engine.py`)
- **WarpEngine**: OpenCVベースの高速画像変換エンジン。ホモグラフィ同期や低解像度グリッドからのマップ生成を担当。
- **PyTorchWarpEngine**: GPU/並列演算を活用した高度なエンジン。AI格子検出（V2）、物理演算ベースのメッシュ最適化（Smooth Warp）を統括。
- **SuperResModel**: PyTorchで実装されたCNNベースの超解像モデル。補正によって失われがちな鮮明度をニューラルネットワークで復元。

##### Controller レイヤー (`controller.py`)
- **Controller**: アプリのライフサイクルとイベントフローを管理。View（入力） -> Model（更新） -> Engine（演算） -> View（表示）の一連の同期を制御し、データの一貫性を保証する。

##### View レイヤー (`view.py`)
- **MainWindow**: 各UIコントロールの配置と、値をControllerが安全に取得・設定するための抽象化メソッドを提供。
- **Canvas**: メイン画像と格子メッシュを重畳描画。マウス操作による制御点移動を検知し、正規化座標をSignalとして発行。
- **PreviewWindow**: 補正完了後の画像をリアルタイムに確認するための専用ウィンドウ。
- **CameraDialog**: 接続されたカメラからリアルタイムプレビューを表示し、任意のタイミングで画像をキャプチャしてControllerへ渡す。

### 5. 既知の未解決課題 (Known Open Issues)

<!-- Issue: PyTorch デバイス（CPU/GPU）の動的な切り替えUI, Status: 保留, Rationale: 現在は自動選択。将来的にユーザーが選択できるように拡張予定。 -->
