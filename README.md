# YOLO ONNX Web (Next.js + onnxruntime-web)

ブラウザだけで YOLO ONNX 推論を行う最小デモです。

- Next.js (App Router) + TypeScript
- onnxruntime-web (WASM)
- 画像1枚アップロードして物体検出
- バウンディングボックス + ラベル + スコア描画
- `scoreThreshold` / `iouThreshold` をUIで変更
- NMSはJavaScript実装

## ファイル構成

```txt
.
├── app/
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── lib/
│   └── yolo.ts
├── public/
│   └── models/
│       ├── coco_labels.json
│       └── yolov8n.onnx   # 自分で配置
├── .eslintrc.json
├── .gitignore
├── next-env.d.ts
├── next.config.mjs
├── package.json
├── README.md
└── tsconfig.json
```

## セットアップ

```bash
npm install
npm run dev
```

- ブラウザで `http://localhost:3000` を開く

## モデル配置

このサンプルは `app/page.tsx` で以下を参照します。

- モデル: `/models/yolov8n.onnx`
- ラベル: `/models/coco_labels.json`

つまり実ファイルは次に置きます。

- `public/models/yolov8n.onnx`
- `public/models/coco_labels.json`（同梱済み）

## 推奨モデルと入手方法（軽量）

軽量でブラウザ実行しやすいのは `yolov8n` クラスです。

### UltralyticsからONNXを作る例

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, opset=12)
```

生成された `yolov8n.onnx` を `public/models/` に配置してください。

## 前処理・後処理の仕様

### 前処理（Step1: 最小構成）

- 入力画像を `640x640` に単純リサイズ
- RGB化して `0-1` 正規化
- `[1, 3, 640, 640]` の `float32` テンソルを作成

### 後処理

- スコア閾値: `scoreThreshold`
- IoU閾値: `iouThreshold`
- NMS: JavaScript実装（クラスごと）
- 出力座標を元画像サイズにスケールバックして描画

## 想定する出力テンソル形状（モデル依存）

この実装は次を優先対応しています。

- `[1, 84, 8400]`（YOLOv8系でよくある）
- `[1, 8400, 84]`
- `85`次元系（`[x,y,w,h,obj,classes...]`）

内部では `session.outputNames[0]` を使用し、`dims` から自動で解釈します。

### 形状が違う場合の確認方法

`lib/yolo.ts` の `postprocess()` 冒頭で `output.dims` を確認してください。
必要に応じて以下を調整します。

- `features-first / preds-first` の判定
- `classStart`（`4` or `5`）
- スコア計算（`bestClassScore` or `objectness * classScore`）

## ブラウザ(WASM)運用の注意点

- 初回モデルロードは重い（数秒かかることがある）
- モデルサイズが大きいほどメモリ消費と待ち時間が増える
- 端末スペックによって推論時間が大きく変わる
- モバイルでは特に軽量モデル推奨

## 実装ポイント

- セッションは `InferenceSession.create()` を1回だけ実行して再利用
- 処理分割
  - `loadModel()`
  - `preprocess()`
  - `runInference()`
  - `postprocess()`
  - `drawDetections()`
- すべてクライアントサイド（サーバー推論なし）

## Step2（任意拡張）

現状は単純リサイズです。精度改善したい場合はレターボックス前処理に変更してください。

- アスペクト比維持で余白を入れて640x640化
- 後処理で余白分を差し引いて座標復元

