# src/reporting/report_generator.py
from pathlib import Path
import pandas as pd
from datetime import datetime

class ReportGenerator:
    """
    Генератор отчётов по результатам детекции табличек.
    Сохраняет отчёт в виде CSV файла с информацией о каждом изображении и результатах детекции.
    Столбцы отчёта:
    - image_path: путь к исходному изображению
    - success: флаг успешной детекции (True/False)
    - confidence: confidence детекции (float)
    - crop_path: путь к сохранённому кропу (если сохранён)
    """
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_detection_report(self, results: list):
        df = pd.DataFrame(results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        csv_path = self.output_dir / f"detection_report_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        
        print(f"Отчёт сохранён: {csv_path}")
        print(f"Успешно детектировано: {sum(1 for r in results if r['success'])} из {len(results)}")