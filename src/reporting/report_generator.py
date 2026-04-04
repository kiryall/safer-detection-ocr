# src/reporting/report_generator.py

from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any

from src.utils.logger import setup_logging

logger = setup_logging(log_file="report.log")


class ReportGenerator:
    """
    Генератор отчётов по результатам обработки.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_report(self, results: List[Dict[str, Any]]):
        """Сохраняет подробный CSV-отчёт."""
        if not results:
            logger.warning("Нет результатов для сохранения отчёта")
            return

        df = pd.DataFrame(results)

        # Красивые названия колонок
        column_rename = {
            "filename": "Имя_файла",
            "original_path": "Путь_к_оригиналу",
            "detection_success": "Детекция_успешна",
            "detection_confidence": "Уверенность_детекции",
            "recognized_text": "Распознанный_текст",
            "text_confidence": "Уверенность_OCR",
            "clean_text": "Чистый_номер",
            "final_name": "Новое_имя_файла",
            "status": "Статус",
            "message": "Сообщение",
        }

        df = df.rename(columns=column_rename)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Основной CSV отчёт
        report_path = self.output_dir / f"sefer_report_{timestamp}.csv"
        df.to_csv(report_path, index=False, encoding="utf-8-sig")

        # Краткий summary
        summary_path = self.output_dir / f"sefer_summary_{timestamp}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Отчёт Sefer Vision — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Всего обработано: {len(df)}\n")
            f.write(f"Успешно: {len(df[df['Статус'] == 'success'])}\n")
            f.write(f"Частично: {len(df[df['Статус'] == 'partial'])}\n")
            f.write(f"Неудачно: {len(df[df['Статус'] == 'failed'])}\n\n")
            f.write(f"Файл с полным отчётом: {report_path.name}\n")

        logger.info(f"Отчёт сохранён: {report_path}")
        logger.info(f"Сводка сохранена: {summary_path}")

        # Вывод в консоль
        success_rate = len(df[df['Статус'] == 'success']) / len(df) * 100 if len(df) > 0 else 0
        print(f"\n✅ Отчёт готов: {report_path.name}")
        print(f"   Успешно: {success_rate:.1f}% ({len(df[df['Статус'] == 'success'])} из {len(df)})")