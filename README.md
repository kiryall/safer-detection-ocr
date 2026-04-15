## structure

```
safer/
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
├── uv.lock
│
├── configs/
│   ├── config.py
│   ├── model_config_aug.yaml
│   └── model_config_no_aug.yaml
│
├── models/
│   └── yolo/
│       ├── yolov8n.pt
│       └── v1/
│           └── best.pt
│
├── notebooks/
│
├── output/
│   ├── results.csv
│   └── visualizations/ (JPG files)
│
├── scripts/
│   ├── run_pipeline.py
│   └── train_yolo.py
│
├── src/
│   ├── detection/
│   │   └── yolo_detector.py
│   ├── grouping/
│   │   └── exif_grouper.py
│   ├── ocr/
│   │   └── ocr_engine.py
│   ├── pipeline/
│   │   ├── main_pipeline.py
│   │   └── batch_processor.py
│   ├── postprocessing/
│   │   ├── text_cleaner.py
│   │   └── validator.py
│   ├── preprocessing/
│   │   ├── image_loader.py
│   │   └── image_utils.py
│   ├── renaming/
│   │   └── renamer.py
│   ├── reporting/
│   │   └── report_generator.py
│   └── utils/
│       ├── data_utils.py
│       ├── logger.py
│       └── yaml_loader.py
│
├── tests/
│   └── test_pipeline.py
│
└── ui/
    └── streamlit_app.py
```