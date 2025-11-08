# Python OCR プロジェクト

OpenCVとTesseractを使用して画像から文字を抽出するPythonアプリケーションです。

## 機能

- 画像ファイルからのテキスト抽出
- 複数の画像形式サポート (PNG, JPG, JPEG, TIFF, BMP, GIF)
- 前処理による認識精度の向上
- 信頼度スコア付きテキスト抽出
- コマンドラインインターフェース（CLI）
- プログラマティックAPI

## 要件

### システム要件
- Python 3.8以上
- Tesseract OCRエンジン

### macOSでのTesseractインストール
```bash
brew install tesseract
# 日本語パックも含める場合
brew install tesseract-lang
```

### Ubuntu/Debianでのインストール
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-jpn  # 日本語パック
```

## セットアップ

1. **リポジトリのクローンまたはダウンロード**

2. **Python仮想環境の作成と有効化**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# または
.venv\\Scripts\\activate  # Windows
```

3. **依存関係のインストール**
```bash
pip install -r requirements.txt
```

## 使用方法

### コマンドライン使用例

基本的な使用方法：
```bash
python src/ocr_processor.py path/to/image.jpg
```

オプション付きの実行：
```bash
# 前処理なしで実行
python src/ocr_processor.py image.png --no-preprocess

# 英語のみで処理
python src/ocr_processor.py image.jpg --lang eng

# 信頼度スコアを表示
python src/ocr_processor.py image.png --confidence

# 結果をファイルに保存
python src/ocr_processor.py image.jpg --output result.txt
```

### プログラマティック使用例

```python
from pathlib import Path
from src.ocr_processor import OCRProcessor

# OCRプロセッサーを初期化
ocr = OCRProcessor(lang='jpn+eng')

# テキストを抽出
image_path = Path('path/to/your/image.jpg')
text = ocr.extract_text(image_path)
print(text)

# 信頼度スコア付きで抽出
text, confidence = ocr.get_text_with_confidence(image_path)
print(f"テキスト: {text}")
print(f"信頼度: {confidence:.1f}%")
```

### デモの実行

```bash
python src/demo.py
```

## テスト

テストを実行するには：
```bash
python -m pytest tests/ -v
```

## プロジェクト構造

```
.
├── .github/
│   └── copilot-instructions.md  # GitHub Copilot指示書
├── src/
│   ├── ocr_processor.py         # メインのOCRプロセッサー
│   └── demo.py                  # デモスクリプト
├── images/
│   └── README.md                # 画像用のプレースホルダー
├── tests/
│   └── test_ocr_processor.py    # テストファイル
├── requirements.txt             # Python依存関係
└── README.md                    # このファイル
```

## サポートされている画像形式

- PNG (.png)
- JPEG (.jpg, .jpeg) 
- TIFF (.tiff)
- BMP (.bmp)
- GIF (.gif)

## OCR精度向上のためのヒント

1. **画像品質**
   - 高解像度（300 DPI以上推奨）
   - 明確なコントラスト（黒いテキスト、白い背景）
   - ノイズの少ない画像

2. **前処理オプション**
   - デフォルトで前処理が適用されます
   - `--no-preprocess`オプションで無効化可能

3. **言語設定**
   - 日本語と英語：`jpn+eng`（デフォルト）
   - 英語のみ：`eng`
   - その他の言語：Tesseractでサポートされている言語コードを使用

## 問題解決

### よくある問題

1. **"pytesseract.TesseractNotFoundError"**
   - Tesseractがインストールされていないか、PATHに含まれていません
   - システム要件を参照してTesseractをインストールしてください

2. **認識精度が低い**
   - 画像の品質を向上させてください
   - 適切な言語設定を使用してください
   - 前処理オプションを調整してください

3. **依存関係のエラー**
   - `pip install -r requirements.txt`を実行してください
   - Python仮想環境が有効化されていることを確認してください

## ライセンス

このプロジェクトはオープンソースです。

## 貢献

プルリクエストや問題報告を歓迎します。