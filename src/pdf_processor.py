#!/usr/bin/env python3
"""
PDF専用テキスト抽出ユーティリティ
pdfplumberとpdf2image+Tesseractを使用してPDFから文字を抽出します。
"""

import pytesseract
from pathlib import Path
from typing import Optional, Dict, List
import argparse
import sys
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """PDFからテキストを抽出するプロセッサー"""

    def __init__(self, lang: str = "jpn+eng"):
        """
        PDFプロセッサーを初期化

        Args:
            lang: Tesseractの言語設定（デフォルト: 'jpn+eng'）
        """
        self.lang = lang

    def extract_text(self, pdf_path: Path) -> str:
        """
        PDFからテキストを抽出

        Args:
            pdf_path: PDFファイルのパス

        Returns:
            抽出されたテキスト
        """
        try:
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDFファイルが見つかりません: {pdf_path}")

            if pdf_path.suffix.lower() != ".pdf":
                raise ValueError(f"PDFファイルではありません: {pdf_path.suffix}")

            # まずpdfplumberでテキスト抽出を試みる
            text = ""
            try:
                try:
                    import pdfplumber
                except ImportError:
                    pdfplumber = None

                if pdfplumber:
                    with pdfplumber.open(pdf_path) as pdf:
                        for page_num, page in enumerate(pdf.pages):
                            try:
                                page_text = page.extract_text()
                                logger.debug(
                                    f"pdfplumber 生テキスト (ページ {page_num + 1}): {repr(page_text[:100])}"
                                )
                                if page_text:
                                    text += page_text + "\n"
                            except Exception as e:
                                logger.warning(
                                    f"pdfplumber ページ {page_num + 1}の抽出でエラー: {e}"
                                )

                # もしpdfplumberで抽出できなければPyPDF2で試す（フォールバック）
                if not text:
                    try:
                        import PyPDF2

                        with open(pdf_path, "rb") as f:
                            reader = PyPDF2.PdfReader(f)
                            for page in reader.pages:
                                try:
                                    page_text = page.extract_text()
                                    if page_text:
                                        text += page_text + "\n"
                                except Exception:
                                    continue
                    except Exception as e:
                        logger.warning(f"PyPDF2でのPDF抽出に失敗: {e}")

            except Exception as e:
                logger.warning(f"PDFテキスト抽出の試行中に警告: {e}")

            # 取得したテキストが空の場合は、画像に変換してOCRで抽出する（スキャンPDF対応）
            cleaned_text = "\n".join(
                line.strip() for line in text.split("\n") if line.strip()
            )
            if cleaned_text:
                logger.info(f"PDFテキスト抽出完了: {len(cleaned_text)}文字")
                return cleaned_text

            # フォールバック: pdfを画像化してTesseractでOCR
            try:
                try:
                    from pdf2image import convert_from_path
                except ImportError:
                    convert_from_path = None

                if convert_from_path is None:
                    raise RuntimeError(
                        "pdf2imageが利用できません（インストールしてください）"
                    )

                custom_config = r"--oem 3 --psm 6"
                images = convert_from_path(str(pdf_path), dpi=300)
                ocr_text = ""
                for page_num, img in enumerate(images):
                    logger.info(f"OCR処理中: ページ {page_num + 1}/{len(images)}")
                    page_text = pytesseract.image_to_string(
                        img, lang=self.lang, config=custom_config
                    )
                    if page_text:
                        ocr_text += page_text + "\n"

                cleaned_ocr = "\n".join(
                    line.strip() for line in ocr_text.split("\n") if line.strip()
                )
                logger.info(f"PDF->画像->OCR完了: {len(cleaned_ocr)}文字")
                return cleaned_ocr
            except Exception as e:
                logger.error(f"PDFを画像化してOCRに回す処理でエラー: {e}")
                raise

        except Exception as e:
            logger.error(f"PDF処理中にエラーが発生しました: {e}")
            raise

    def process_pdfs_directory(
        self, pdfs_dir: Path, output_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """
        指定されたディレクトリ内のすべてのPDFを処理

        Args:
            pdfs_dir: PDFが格納されているディレクトリのパス
            output_dir: 結果を保存するディレクトリのパス（None の場合は 'output' ディレクトリを使用）

        Returns:
            {PDFファイル名: 抽出テキスト} の辞書
        """
        try:
            if not pdfs_dir.exists():
                raise FileNotFoundError(f"ディレクトリが見つかりません: {pdfs_dir}")

            # PDFファイルを検索
            pdf_files = []
            for ext in [".pdf", ".PDF"]:
                pdf_files.extend(pdfs_dir.glob(f"*{ext}"))

            if not pdf_files:
                logger.warning("処理可能なPDFファイルが見つかりません。")
                return {}

            logger.info(f"見つかったPDFファイル数: {len(pdf_files)}")

            results = {}

            # 出力ディレクトリの設定（指定されていない場合はデフォルトで 'output' を使用）
            if output_dir is None:
                output_dir = Path("output")

            # 出力ディレクトリの作成
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"出力ディレクトリ: {output_dir}")

            for pdf_file in pdf_files:
                logger.info(f"処理中: {pdf_file.name}")
                try:
                    text = self.extract_text(pdf_file)
                    results[pdf_file.name] = text

                    # テキストファイルとして保存（常に保存）
                    output_file = output_dir / f"{pdf_file.stem}.txt"
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
                    logger.error(f"{pdf_file.name}の処理中にエラー: {e}")
                    results[pdf_file.name] = f"エラー: {e}"

            return results

        except Exception as e:
            logger.error(f"ディレクトリ処理中にエラーが発生しました: {e}")
            raise


def main():
    """メイン関数 - コマンドライン実行時の処理"""
    parser = argparse.ArgumentParser(
        description="PDFからテキストを抽出します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一PDFの処理（デフォルトでoutput/に保存）
  python pdf_processor.py sample.pdf
  python pdf_processor.py document.pdf --lang eng
  
  # ディレクトリ内のすべてのPDFを処理（デフォルトでoutput/に保存）
  python pdf_processor.py pdfs/ --directory
  python pdf_processor.py pdfs/ -d --output results/
        """,
    )

    parser.add_argument(
        "pdf_path", type=str, help="処理するPDFファイルまたはディレクトリのパス"
    )
    parser.add_argument(
        "--lang", default="jpn+eng", help="Tesseractの言語設定 (デフォルト: jpn+eng)"
    )
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
        help="ディレクトリ内のすべてのPDFを処理",
    )

    args = parser.parse_args()

    try:
        # PDFプロセッサーの初期化
        pdf_processor = PDFProcessor(lang=args.lang)
        input_path = Path(args.pdf_path)

        print(f"言語設定: {args.lang}")

        # ディレクトリ処理の場合
        if args.directory or input_path.is_dir():
            print(f"ディレクトリを処理中: {input_path}")

            output_dir = Path(args.output) if args.output else None
            results = pdf_processor.process_pdfs_directory(input_path, output_dir)

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
                print("処理可能なPDFファイルが見つかりませんでした。")

        # 単一ファイル処理の場合
        else:
            print(f"PDFを処理中: {input_path}")

            text = pdf_processor.extract_text(input_path)

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
