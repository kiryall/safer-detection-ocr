# scripts/run_pipeline.py

from src.pipeline.main_pipeline import run_pipeline
from configs.config import RAW_DATA_DIR, OUTPUT_DIR

if __name__ == "__main__":
    run_pipeline(input_dir=RAW_DATA_DIR, output_dir=OUTPUT_DIR)