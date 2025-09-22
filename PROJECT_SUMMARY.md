# MVP Project Summary

## Project Overview

MVP (Motion Vector Propagation) is a zero-shot object detection framework that combines motion vector analysis with OWL-ViT for efficient video object detection. The project has been organized and structured for easy deployment and usage.

## Project Structure

```
MVP/
├── src/                    # Core source code
│   ├── mvp_detector.py    # Main MVP detector class
│   ├── motion_analyzer.py # Motion vector analysis
│   ├── owl_detector.py    # OWL-ViT integration
│   ├── main.py            # Command line interface
│   └── __init__.py        # Package initialization
├── configs/               # Configuration files
│   ├── default.yaml      # Default configuration
│   └── imagenet_vid.yaml # ImageNet VID specific config
├── evaluation/            # Evaluation scripts
│   ├── evaluate.py       # Main evaluation script
│   ├── metrics.py        # Evaluation metrics
│   ├── visualization.py  # Visualization utilities
│   └── __init__.py       # Package initialization
├── utils/                 # Utility functions
│   ├── extract_motion_vectors.py
│   ├── visualization.py
│   ├── data_utils.py
│   └── __init__.py
├── examples/              # Usage examples
│   ├── basic_usage.py
│   ├── custom_dataset.py
│   └── evaluation_example.py
├── mv-extractor/          # Motion vector extraction tool
├── docs/                  # Documentation
├── README.md              # Main documentation
├── QUICKSTART.md          # Quick start guide
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── install.sh            # Linux/macOS installation script
├── install.bat           # Windows installation script
└── LICENSE               # MIT License
```

## Key Features

1. **Motion Vector Analysis**: Efficient extraction and analysis of motion vectors from compressed videos
2. **OWL-ViT Integration**: Zero-shot object detection using OWL-ViT
3. **Smart Propagation**: 9-grid motion vector analysis for accurate object tracking
4. **Adaptive Strategy**: Dynamic switching between detection and propagation modes
5. **Performance Optimization**: Significant speedup compared to frame-wise detection

## Technical Implementation

### Core Components

1. **MVPDetector**: Main orchestrator class that combines motion vector analysis with OWL-ViT detection
2. **MotionAnalyzer**: Handles motion vector analysis and object propagation
3. **OWLDetector**: Wrapper for OWL-ViT zero-shot object detection
4. **Evaluation Tools**: Comprehensive evaluation metrics and visualization

### Key Algorithms

1. **9-Grid Motion Analysis**: Divides bounding boxes into 3x3 grids and analyzes motion vectors within each cell
2. **Translation Detection**: Identifies pure translation motion patterns
3. **Scaling Detection**: Identifies pure scaling motion patterns
4. **Area Scaling Check**: Monitors for significant changes in object area to trigger re-detection

## Usage Patterns

### Command Line Interface

```bash
# Single video processing
python src/main.py --video_path video.mp4 --mv_dir mvs/video_name --output_dir results/

# Batch processing
python src/main.py --video_dir videos/ --mv_root mvs/ --output_root results/

# Evaluation
python evaluation/evaluate.py --pred_dir results/ --gt_dir ground_truth/ --output_dir eval_results/
```

### Python API

```python
from src.mvp_detector import MVPDetector

detector = MVPDetector()
stats = detector.process_video(video_path, mv_dir, output_dir)
```

## Performance Results

Based on the original implementation, MVP achieves:

- **mAP@0.2**: 0.747
- **mAP@0.3**: 0.721
- **mAP@0.5**: 0.609
- **mAP@[0.5:0.95]**: 0.316

With significant speedup compared to frame-wise detection methods.

## Installation and Setup

1. **Automatic Installation**: Use `install.sh` (Linux/macOS) or `install.bat` (Windows)
2. **Manual Installation**: Follow the steps in `QUICKSTART.md`
3. **Dependencies**: All required packages are listed in `requirements.txt`

## Documentation

- **README.md**: Comprehensive project documentation
- **QUICKSTART.md**: Quick start guide for new users
- **Examples**: Working examples in the `examples/` directory
- **Configuration**: YAML configuration files for different use cases

## Contributing

The project is structured to be easily extensible:

1. **Modular Design**: Each component is in its own module
2. **Configuration-Driven**: Behavior can be customized via YAML files
3. **Well-Documented**: Comprehensive documentation and examples
4. **Tested**: Includes evaluation tools and metrics

## Future Enhancements

Potential areas for improvement:

1. **Additional Motion Vector Extractors**: Support for more video formats
2. **Advanced Motion Models**: More sophisticated motion analysis
3. **Real-time Processing**: Optimization for real-time applications
4. **Multi-object Tracking**: Enhanced tracking for multiple objects
5. **Custom Datasets**: Better support for custom datasets

## Citation

If you use this code in your research, please cite the ICASSP 2025 paper:

```bibtex
@inproceedings{mvp2025,
  title={MVP: Motion Vector Propagation for Zero-Shot Object Detection},
  author={Binhua Huang, Co-authors},
  booktitle={ICASSP 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
