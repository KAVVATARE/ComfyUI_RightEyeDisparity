# ComfyUI_RightEyeDisparity

VR動画用の右目視差動画を生成するシンプルなComfyUIカスタムノードです。左目動画と深度マップから右目動画のみを効率的に生成します。

## 特徴

- 🎬 **動画バッチ処理対応**: メモリ効率を考慮した設計
- 👁️ **右目動画のみ出力**: 必要最小限の処理でメモリを節約
- 🔧 **高品質な視差生成**: ComfyStereoの実績あるアルゴリズムを使用
- 📹 **既存ワークフローとの統合**: Video CombineやMeta Batchノードと連携

## インストール

1. ComfyUIのカスタムノードフォルダにクローン：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI_RightEyeDisparity
```

2. 依存関係をインストール：
```bash
cd ComfyUI_RightEyeDisparity
pip install -r requirements.txt
```

## 使用方法

### Video Right Eye Disparity ノード

**入力:**
- `images`: 左目用の動画フレーム（バッチ）
- `depth_maps`: 各フレームの深度マップ
- `fill_technique`: 視差生成時の塗りつぶし技術

**出力:**
- `images`: 生成された右目動画フレーム

**主要パラメータ:**
- `divergence` (0.05-15): 視差の強さ。大きいほど立体感が強い
- `separation` (-5-5): 水平オフセット調整
- `stereo_balance` (-0.95-0.95): 左右の視差バランス
- `fill_technique`: 推奨は "Fill - Polylines Soft"

### ワークフローの例

1. **Load Video** → 左目動画を読み込み
2. **MiDaS Depth Map** → 深度マップを生成
3. **Video Right Eye Disparity** → 右目動画を生成
4. **Upscale** → 必要に応じて左右個別にアップスケール
5. **Video Combine** → Meta Batchで動画として保存

## メモリ最適化のヒント

- バッチサイズを調整してメモリ使用量をコントロール
- `direction_aware_depth_blur`は無効化されているため高速
- 深度マップの返却を省略してメモリを節約

## トラブルシューティング

### メモリ不足
- より小さいバッチサイズで処理
- 解像度を下げて処理後にアップスケール

### 視差が不自然
- `divergence`を2-5の範囲で調整
- 深度マップの品質を確認

## ライセンス

MIT License

## 謝辞

このプロジェクトは[ComfyStereo](https://github.com/Dobidop/ComfyStereo)のstereoimage_generation.pyを使用しています。
