# Quick Start Guide

This guide will help you get started with MVP detector quickly.

## Installation

### Option 1: Using the installation script (Recommended)

**Linux/macOS:**
```bash
chmod +x install.sh
./install.sh
```

**Windows:**
```cmd
install.bat
```

### Option 2: Manual installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv mvp_env
source mvp_env/bin/activate  # Linux/macOS
# or
mvp_env\Scripts\activate.bat  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install motion vector extractor:
```bash
cd mv-extractor
pip install -e .
cd ..
```

4. Install MVP package:
```bash
pip install -e .
```

## Basic Usage

### 1. Extract Motion Vectors

First, you need to extract motion vectors from your videos:

```bash
# For a single video
python utils/extract_motion_vectors.py --video_path path/to/video.mp4 --output_dir path/to/mvs

# For multiple videos
python utils/extract_motion_vectors.py --video_dir path/to/videos --output_dir path/to/mvs
```

### 2. Run Detection

```bash
# Single video
python src/main.py --video_path path/to/video.mp4 --mv_dir path/to/mvs/video_name --output_dir results/

# Multiple videos
python src/main.py --video_dir path/to/videos --mv_root path/to/mvs --output_root results/
```

### 3. Evaluate Results

```bash
python evaluation/evaluate.py --pred_dir results/ --gt_dir path/to/ground_truth --output_dir evaluation_results/
```

## Python API Usage

```python
from src.mvp_detector import MVPDetector

# Initialize detector
detector = MVPDetector(
    model_id="google/owlv2-large-patch14-ensemble",
    device="cuda",
    confidence_threshold=0.5
)

# Process video
stats = detector.process_video(
    video_path="path/to/video.mp4",
    motion_vector_dir="path/to/motion_vectors/video_name",
    output_dir="path/to/output"
)

print(f"Processing completed: {stats['fps']:.2f} FPS")
```

## Configuration

You can customize the detector behavior by modifying the configuration files in `configs/`:

- `default.yaml`: Default configuration
- `imagenet_vid.yaml`: ImageNet VID specific configuration

## Examples

Check the `examples/` directory for more detailed usage examples:

- `basic_usage.py`: Basic usage example
- `custom_dataset.py`: Custom dataset processing
- `evaluation_example.py`: Evaluation example

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Motion vectors not found**: Make sure to extract motion vectors first
3. **Import errors**: Make sure all dependencies are installed correctly

### Getting Help

- Check the README.md for detailed documentation
- Look at the examples in the `examples/` directory
- Open an issue on GitHub if you encounter problems

## Next Steps

1. Try the basic usage example
2. Process your own videos
3. Experiment with different configurations
4. Evaluate on your dataset
5. Contribute to the project!
