from pathlib import Path
from src.pipeline.main_pipeline import SaferPipeline


pipeline = SaferPipeline()
pipeline.process_single_image(image_path=Path("tests\MDZ089 _1.JPG"))