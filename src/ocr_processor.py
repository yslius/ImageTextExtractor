#!/usr/bin/env python3
"""
OCR画像テキスト抽出ユーティリティ
OpenCVとTesseractを使用して画像から文字を抽出します。
"""

import cv2
import pytesseract
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import argparse
from PIL import Image
import sys
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """画像からテキストを抽出するOCRプロセッサー"""

    # サポートされている画像形式
    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}

    def __init__(self, lang: str = "jpn+eng"):
        """
        OCRプロセッサーを初期化

        Args:
            lang: Tesseractの言語設定（デフォルト: 'jpn+eng'）
        """
        self.lang = lang

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        OCR精度向上のための画像前処理

        Args:
            image_path: 処理する画像のパス

        Returns:
            前処理された画像（numpy配列）
        """
        try:
            # 画像を読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"画像を読み込めませんでした: {image_path}")

            # グレースケール変換
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # ノイズ除去
            denoised = cv2.medianBlur(gray, 5)

            # 適応的閾値処理で二値化
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            return thresh

        except Exception as e:
            logger.error(f"画像前処理中にエラーが発生しました: {e}")
            raise

    def extract_text(self, image_path: Path, preprocess: bool = True) -> str:
        """
        画像からテキストを抽出

        Args:
            image_path: 画像ファイルのパス
            preprocess: 前処理を行うかどうか

        Returns:
            抽出されたテキスト
        """
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

            if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"サポートされていない画像形式です: {image_path.suffix}"
                )

            if preprocess:
                # 前処理を適用
                processed_image = self.preprocess_image(image_path)
                # OpenCVからPILに変換
                pil_image = Image.fromarray(processed_image)
            else:
                # 直接画像を読み込み
                pil_image = Image.open(image_path)

            # Tesseract設定
            custom_config = r"--oem 3 --psm 6"

            # テキスト抽出
            text = pytesseract.image_to_string(
                pil_image, lang=self.lang, config=custom_config
            )

            # 空白文字の整理
            cleaned_text = "\n".join(
                line.strip() for line in text.split("\n") if line.strip()
            )

            logger.info(f"テキスト抽出完了: {len(cleaned_text)}文字")
            return cleaned_text

        except Exception as e:
            logger.error(f"テキスト抽出中にエラーが発生しました: {e}")
            raise

    def get_text_with_confidence(self, image_path: Path) -> Tuple[str, float]:
        """
        テキストと信頼度スコアを取得

        Args:
            image_path: 画像ファイルのパス

        Returns:
            (抽出テキスト, 平均信頼度スコア)のタプル
        """
        try:
            processed_image = self.preprocess_image(image_path)
            pil_image = Image.fromarray(processed_image)

            # 詳細データを取得
            data = pytesseract.image_to_data(
                pil_image, lang=self.lang, output_type=pytesseract.Output.DICT
            )

            # 信頼度スコアの計算
            confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # テキストの抽出
            text = pytesseract.image_to_string(pil_image, lang=self.lang)
            cleaned_text = "\n".join(
                line.strip() for line in text.split("\n") if line.strip()
            )

            return cleaned_text, avg_confidence

        except Exception as e:
            logger.error(f"信頼度付きテキスト抽出中にエラーが発生しました: {e}")
            raise

    def process_images_directory(
        self, images_dir: Path, output_dir: Optional[Path] = None
    ) -> dict:
        """
        指定されたディレクトリ内のすべての画像を処理

        Args:
            images_dir: 画像が格納されているディレクトリのパス
            output_dir: 結果を保存するディレクトリのパス（None の場合は 'output' ディレクトリを使用）

        Returns:
            {画像ファイル名: 抽出テキスト} の辞書
        """
        try:
            if not images_dir.exists():
                raise FileNotFoundError(f"ディレクトリが見つかりません: {images_dir}")

            # 画像ファイルを検索
            image_files = []
            for ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
                image_files.extend(images_dir.glob(f"*{ext}"))
                image_files.extend(images_dir.glob(f"*{ext.upper()}"))

            if not image_files:
                logger.warning("処理可能な画像ファイルが見つかりません。")
                return {}

            logger.info(f"見つかった画像ファイル数: {len(image_files)}")

            results = {}

            # 出力ディレクトリの設定（指定されていない場合はデフォルトで 'output' を使用）
            if output_dir is None:
                output_dir = Path("output")

            # 出力ディレクトリの作成
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"出力ディレクトリ: {output_dir}")

            for image_file in image_files:
                logger.info(f"処理中: {image_file.name}")
                try:
                    text = self.extract_text(image_file)
                    results[image_file.name] = text

                    # テキストファイルとして保存（常に保存）
                    output_file = output_dir / f"{image_file.stem}.txt"
                    if text:
                        output_file.write_text(text, encoding="utf-8")
                        logger.info(f"結果を保存しました: {output_file}")
                    else:
                        # テキストが空の場合でも空ファイルを作成
                        output_file.write_text(
                            "テキストが検出されませんでした。", encoding="utf-8"
                        )
                        logger.info(f"空の結果を保存しました: {output_file}")

                except Exception as e:
                    logger.error(f"{image_file.name}の処理中にエラー: {e}")
                    results[image_file.name] = f"エラー: {e}"

            return results

        except Exception as e:
            logger.error(f"ディレクトリ処理中にエラーが発生しました: {e}")
            raise


def main():
    """メイン関数 - コマンドライン実行時の処理"""
    parser = argparse.ArgumentParser(
        description="OCRを使用して画像からテキストを抽出します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一画像の処理（デフォルトでoutput/に保存）
  python ocr_processor.py image.jpg
  python ocr_processor.py image.png --no-preprocess
  python ocr_processor.py image.tiff --lang eng --confidence
  
  # ディレクトリ内のすべての画像を処理（デフォルトでoutput/に保存）
  python ocr_processor.py images/ --directory
  python ocr_processor.py images/ -d --output results/
        """,
    )

    parser.add_argument(
        "image_path", type=str, help="処理する画像ファイルまたはディレクトリのパス"
    )
    parser.add_argument(
        "--lang", default="jpn+eng", help="Tesseractの言語設定 (デフォルト: jpn+eng)"
    )
    parser.add_argument(
        "--no-preprocess", action="store_true", help="画像前処理をスキップ"
    )
    parser.add_argument("--confidence", action="store_true", help="信頼度スコアを表示")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="結果を保存するファイルまたはディレクトリのパス",
    )
    parser.add_argument(
        "--directory",
        "-d",
        action="store_true",
        help="ディレクトリ内のすべての画像を処理",
    )

    args = parser.parse_args()

    try:
        # OCRプロセッサーの初期化
        ocr = OCRProcessor(lang=args.lang)
        input_path = Path(args.image_path)

        print(f"言語設定: {args.lang}")

        # ディレクトリ処理の場合
        if args.directory or input_path.is_dir():
            print(f"ディレクトリを処理中: {input_path}")

            output_dir = Path(args.output) if args.output else None
            results = ocr.process_images_directory(input_path, output_dir)

            if results:
                print("\\n=== 処理結果 ===")
                for filename, text in results.items():
                    print(f"\\n--- {filename} ---")
                    if text.startswith("エラー:"):
                        print(text)
                    elif text:
                        # テキストが長い場合は最初の300文字のみ表示
                        if len(text) > 300:
                            print(f"{text[:300]}...\\n[文字数: {len(text)}文字]")
                        else:
                            print(f"{text}\\n[文字数: {len(text)}文字]")
                    else:
                        print("テキストが検出されませんでした。")
            else:
                print("処理可能な画像ファイルが見つかりませんでした。")

        # 単一ファイル処理の場合
        else:
            print(f"画像を処理中: {input_path}")

            if args.confidence:
                # 信頼度付きで抽出
                text, confidence = ocr.get_text_with_confidence(input_path)
                print(f"\\n信頼度スコア: {confidence:.1f}%")
            else:
                # 通常の抽出
                text = ocr.extract_text(input_path, preprocess=not args.no_preprocess)

            if text:
                print("\\n=== 抽出されたテキスト ===")
                print(text)

                # ファイルに保存（デフォルトでoutputディレクトリに保存）
                if args.output:
                    output_path = Path(args.output)
                else:
                    output_dir = Path("output")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{input_path.stem}.txt"

                output_path.write_text(text, encoding="utf-8")
                print(f"\\n結果を保存しました: {output_path}")
            else:
                print("テキストが検出されませんでした。")

                # テキストが空の場合でも空ファイルを作成
                if args.output:
                    output_path = Path(args.output)
                else:
                    output_dir = Path("output")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{input_path.stem}.txt"

                output_path.write_text(
                    "テキストが検出されませんでした。", encoding="utf-8"
                )
                print(f"\\n空の結果を保存しました: {output_path}")

    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
