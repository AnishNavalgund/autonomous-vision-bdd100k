# Autonomous Vision BDD100K

End-to-end object detection pipeline using BDD100K dataset for autonomous driving applications.

## Overview

This project implements a complete computer vision pipeline for autonomous driving using the Berkeley DeepDrive (BDD100K) dataset. It provides a comprehensive workflow from raw data parsing to model training, and evaluation.

## Project Flow

The project follows this sequential workflow:

1. **Data Ingestion**: Raw BDD100K JSON labels and images
2. **Data Parsing**: Convert BDD100K format to structured CSV/Parquet files
3. **Format Conversion**: Transform BDD100K annotations to COCO format for custom training pipeline
4. **YOLO Preparation**: Convert COCO to YOLO format for training
5. **Model Training**: Train YOLO models on prepared data
6. **Analysis**: Quantitative and qualitative analysis with overlay visualization

## Complete Project Structure

```
autonomous-vision-bdd100k/
├── README.md                                
├── docs/                                    # Documentation
│   ├── 01_data_analysis.md                  # BDD100K dataset analysis report 
│   ├── 02_model_selection.md                # Model Selection, Architecture and Training Settings
│   ├── 03_model_analysis.md                 # Mectric Selection, Quantitative, Qualitative Analysis and Improvement areas
│   └── images/                              # Documentation images
├── data/                                    # Dataset storage and processing
│   └── parsed_data/                         # Parsed CSV/Parquet files (Will be created by parser_core.py also committed)
├── src/autonomous_vision/                   # Core Python package
│   ├── config.py                            # Paths and constants
│   ├── data_parser/                         # Data parsing and conversion
│   │   ├── __init__.py
│   │   ├── parser_core.py                   # BDD to CSV/Parquet format converter
│   │   ├── parsing_logic.py                 # Parsing orchestrator
│   │   └── bdd_to_coco.py                   # BDD to COCO format converter
│   ├── object_detection/                    # YOLO training and inference
│   │   ├── train_yolo.py                    # Main YOLO training and Eval script
│   │   ├── data_loader.py                   # COCO data loading utities
│   │   ├── label_utils.py                   # YOLO label creator
│   │   └── yolo_overlay.py                  # YOLO inference and visualization
│   └── utils/                               # Utility functions
├── notebooks/                               
│   ├── DataAnalysis/                        # Exploratory data analysis
│   │   ├── 01_EDA_RawData.ipynb             # Raw data exploration
│   │   └── 02_BDD100k_Analysis.ipynb        # In-depth BDD100K dataset analysis
│   ├── 01_CustomLoader_FasterRCNN.ipynb     # Custom data loader implementation
│   ├── 02_YOLO_QuantitativeAnalysis.ipynb   # YOLO model quantitative evaluation
│   ├── 03_YOLO_QualitativeAnalysis.ipynb    # YOLO model qualitative evaluation
│   ├── test.ipynb                           # Testing and experimentation
├── .pre-commit-config.yaml                  # Pre-commit hooks configuration
├── .gitignore                               # Git ignore patterns
├── Dockerfile                               # Docker container definition
├── docker-compose.yml                       # Docker Compose configuration
├── docker_requirements.txt                  # Docker requirements file
├── pyproject.toml                           # Project configuration and dependencies                        
└── uv.lock                                  # Dependency lock file

```

## Requirements

### System Requirements
- Python 3.12+
- uv (for dependency management)

## Installation and Setup

### Clone the repository

```bash
git clone https://github.com/anishnavalgund/autonomous-vision-bdd100k.git
cd autonomous-vision-bdd100k
```

### Data Preparation

```bash
# Structure should be:
data/
├── raw_bdd_jsons/
│   ├── bdd100k_labels_images_train.json
│   └── bdd100k_labels_images_val.json
├── yolo_data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── parsed_data/
    ├── train_data.parquet
    └── val_data.parquet
```
Note: parquet files are created by the pipeline but its commited to the repository for docker build and data analysis (without running pipeline)

## Data Analysis via Docker

```bash
docker compose up --build -d
```
Note: This will start a jupyter notebook server at `http://localhost:8888`


## Running the pipeline

### 1. Install dependencies

```bash
uv sync
source .venv/bin/activate
```
### 2. Pre-commit hooks 

```bash
pre-commit install
```

### 3. Pipeline - Create data

#### Data Conversion

```bash
uv run python -m autonomous_vision.data_parser.parsing_logic
```

#### Start YOLO Training

Note: Make sure paths and settings are correct in `config.py`

```bash
uv run python -m autonomous_vision.object_detection.train_yolo
```

#### YOLO Inference

```bash
uv run python -m autonomous_vision.object_detection.yolo_overlay
```
Overlay images will be saved on the disk.

## Documentation

1. [Exploratory data analysis](docs/01_data_analysis.md) 
2. [Model Selection, Architecture and Training Settings](docs/02_model_selection.md) 
3. [Mectric Selection, Quantitative, Qualitative Analysis and Improvement areas](docs/03_model_analysis.md) 


