# XCR: Extended Kernel Retinex Implementation

## 概要 (Overview)

本リポジトリは、低照度画像強調のための新しいアプローチ **XCR (Extended/Exponential Kernel Retinex)** のPython実装です。
従来のガウシアンカーネルの代わりに、シグモイド関数や複素指数関数に基づいた**非線形分離型フィルタ (Sigmoid/XCR Separable Blur)** を採用することで、計算コストを抑えつつ効果的な照明成分の推定を実現しています。

本プロジェクトには、Webカメラでのリアルタイムデモと、既存手法（MSR, LIME-like）との定量比較ベンチマークが含まれています。

## 参考文献 (Reference)

本実装は、以下の研究成果に基づいています。

- **論文**: Extended Kernel Retinex (XCR) に基づく低照度画像強調手法の検討  
- **著者**: 奥河 董馬 (Toma Okugawa)  
- **DOI**: [10.51094/jxiv.1961](https://doi.org/10.51094/jxiv.1961)

## ファイル構成 (File Structure)

本コードを実行するには、以下のディレクトリ構成が必要です。  
`dataset` フォルダに評価用画像を配置してください。

```text
.
├── main1.py           # [Demo] Webカメラによるリアルタイム強調 (出力: screenshots/)
├── main2.py           # [Benchmark] 定量評価と画像比較 (入力: dataset/ -> 出力: results/)
├── requirements.txt   # 依存ライブラリ一覧
├── LICENSE            # MIT License
├── README.md          # 本ドキュメント
├── screenshots/       # (自動生成) main1.py のスクリーンショット保存先
├── results/           # (自動生成) main2.py の比較画像保存先
└── dataset/           # (必須) 評価用データセットフォルダ
    ├── low/           #  <- ここに「入力用の低照度画像」を入れてください
    └── gt/            #  <- ここに「正解画像(Ground Truth)」を入れてください (ファイル名はlowと同じ)

```

## 環境構築 (Installation)

動作には **Python 3.x** が必要です。  
FSSRとは異なり、MediaPipeは使用しません。

```bash
pip install -r requirements.txt
```

## 使用方法 (Usage)

### 1. リアルタイムデモ (`main1.py`)

Webカメラを接続した状態で実行してください。XCRアルゴリズムの効果をライブで確認できます。

```bash
python main1.py
```

**操作**

- `Esc`: 終了  
- `s`: 現在のフレームを `screenshots/` に保存  

### 2. 比較ベンチマーク (`main2.py`)

**準備**:  
`dataset/low` と `dataset/gt` に評価用画像ペア（同じファイル名）を配置してください。

```bash
python main2.py
```

**実行結果**

- コンソールに各手法（XCR, MSR, LIME-like）の **PSNR / SSIM / 処理時間** が表示されます。
- `results/` フォルダに、比較画像（入力・各手法の出力・正解画像の並列比較）が保存されます。

## アルゴリズムの特徴 (Algorithm)

- **Sigmoid Separable Blur**  
  従来のガウシアンブラーよりもエッジ保存特性や計算効率において独自の特性を持つフィルタを使用。
- **Comparison**  
  一般的な MSR (Multi-Scale Retinex) や LIME (Low-light Image Enhancement) 風の手法と比較実装されています。

## Author

奥河 董馬 (Toma Okugawa)  
弓削商船高等専門学校 (National Institute of Technology, Yuge College)
