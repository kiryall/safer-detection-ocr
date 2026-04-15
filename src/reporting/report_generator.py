# src/reporting/report_generator.py

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from colorama import Fore, Style

import pandas as pd

from configs.config import LOGGING_CONSOLE
from src.utils.logger import setup_logging

logger = setup_logging(
    log_file="report.log",
    console=LOGGING_CONSOLE,
    remove_file=True,
    logger_name="report",
    level=logging.DEBUG,
)


class ReportGenerator:
    """
    Генератор отчётов по результатам обработки.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_report(self, results: List[Dict[str, Any]]):
        """Сохраняет подробный CSV-отчёт."""
        if not results:
            logger.warning("Нет результатов для сохранения отчёта")
            return

        df = pd.DataFrame(results)

        # названия колонок
        column_rename = {
            "filename": "Имя_файла",
            "original_path": "Путь_к_оригиналу",
            "detection_confidence": "Уверенность_детекции",
            "recognized_text": "Распознанный_текст",
            "text_confidence": "Уверенность_OCR",
            "clean_text": "Чистый_номер",
            "final_name": "Новое_имя_файла",
            "final_path": "Путь_к_финальному_файлу",
            "status": "Статус",
            "message": "Сообщение",
            'confidence_level': 'Общий_уровень_уверенности'
        }

        df = df.rename(columns=column_rename)

        # Основной CSV отчёт
        report_path = self.output_dir / "sefer_report.csv"
        df.to_csv(report_path, index=False, encoding="utf-8-sig")

        # Краткий summary
        summary_path = self.output_dir / "sefer_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(
                f"Отчёт Sefer Vision — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write("=" * 50 + "\n")
            f.write(f"Всего обработано: {len(df)}\n")
            f.write(f"Успешно: {len(df[df['Статус'] == 'success'])}\n")
            f.write(f"Частично: {len(df[df['Статус'] == 'partial'])}\n")
            f.write(f"Неудачно: {len(df[df['Статус'] == 'failed'])}\n\n")
            f.write(f"Файл с полным отчётом: {report_path.name}\n")

        logger.info(f"Отчёт сохранён: {report_path}")
        logger.info(f"Сводка сохранена: {summary_path}")

        # Вывод в консоль
        total = len(df)
        if total > 0:
            success = len(df[df["Статус"] == "success"])
            low = len(df[df["Статус"] == "low_confidence"])
            failed = total - success - low

            print("\n" + "=" * 90)
            print(f"{Fore.GREEN}📊 ИТОГОВЫЙ ОТЧЁТ{Style.RESET_ALL}")
            print("=" * 90)
            print(f"Файлов обработано     : {total}")
            print(f"{Fore.GREEN}Успешно               : {success} ({success/total*100:.1f}%){Style.RESET_ALL}")
            print(f"{Fore.YELLOW}С сомнением           : {low}{Style.RESET_ALL}")
            print(f"{Fore.RED}Не распознано         : {failed}{Style.RESET_ALL}")
            print("=" * 90)
            print(f"Отчёт: {report_path}")
        else:
            print("\n❌ Нет данных для формирования отчёта.")
