#!/usr/bin/env python3
"""
シンプルなOCRデモスクリプト
"""

from pathlib import Path
import sys
import os

# srcディレクトリをパスに追加
src_path = Path(__file__).parent
sys.path.append(str(src_path))


def demo():
    """OCRのデモンストレーション"""
    try:
        from ocr_processor import OCRProcessor

        print("=== OCR デモンストレーション ===")
        print("Tesseractが正しくインストールされているかテストします...")

        # 簡単なテストイメージがある場合の処理
        images_dir = Path(__file__).parent.parent / "images"

        if not images_dir.exists():
            print(f"画像ディレクトリが見つかりません: {images_dir}")
            print("テスト画像を配置してください。")
            return

        # 画像ファイルを検索
        image_files = []
        for ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            print("処理可能な画像ファイルが見つかりません。")
            print(
                "imagesディレクトリに画像ファイル（PNG、JPG、TIFF、BMPなど）を配置してください。"
            )
            return

        # OCRプロセッサーを初期化
        ocr = OCRProcessor()

        for image_file in image_files[:3]:  # 最初の3つのファイルのみ処理
            print(f"\\n--- 処理中: {image_file.name} ---")
            try:
                text = ocr.extract_text(image_file)
                if text:
                    print(f"抽出テキスト:\\n{text[:200]}...")  # 最初の200文字のみ表示
                else:
                    print("テキストが検出されませんでした。")
            except Exception as e:
                print(f"エラー: {e}")

        print("\\nデモ完了!")

    except ImportError as e:
        print(f"必要なライブラリがインストールされていません: {e}")
        print("pip install -r requirements.txt を実行してください。")
    except Exception as e:
        print(f"予期しないエラー: {e}")


if __name__ == "__main__":
    demo()
