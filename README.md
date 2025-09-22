# MVP: Motion Vector Propagation for Zero-Shot Video Object Detection

A novel approach for efficient zero-shot object detection in videos by leveraging motion vectors from compressed video streams combined with OWL-ViT for object detection.

## 🎯 Overview

MVP (Motion Vector Propagation) is a zero-shot object detection framework that combines:
- **Motion Vector Analysis**: Extracts and analyzes motion vectors from H.264/MPEG-4 compressed videos
- **OWL-ViT Integration**: Uses OWL-ViT for zero-shot object detection on keyframes
- **Intelligent Propagation**: Propagates detections across frames using motion vector analysis
- **Efficient Tracking**: Reduces computational overhead by avoiding full detection on every frame

## ✨ Key Features

- **Motion Vector Processing**: Efficient extraction and analysis of motion vectors from compressed videos
- **Zero-shot Detection**: OWL-ViT based zero-shot object detection capabilities
- **Smart Propagation**: 9-grid motion vector analysis for accurate object tracking
- **Adaptive Strategy**: Dynamic switching between detection and propagation modes
- **Performance Optimization**: Significant speedup compared to frame-wise detection

## 🏗️ Architecture

The framework consists of three main components:

1. **Motion Vector Extractor**: Extracts motion vectors, frame types, and timestamps from compressed videos
2. **OWL-ViT Detector**: Performs zero-shot object detection on keyframes and when needed
3. **Motion Propagation**: Uses 9-grid analysis to propagate detections across frames

## 📋 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
PIL>=8.3.0
ffmpeg-python>=0.2.0
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/microa/MVP.git
cd MVP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install motion vector extractor:
```bash
cd mv-extractor
pip install -e .
```

## 📊 Dataset Preparation

1. Prepare your video dataset in the following structure:
```
data/
├── videos/           # Video files (.mp4)
├── motion_vectors/   # Extracted motion vectors
│   └── video_name/
│       ├── frame_types.txt
│       ├── timestamps.txt
│       └── motion_vectors/
│           ├── mvs-0.npy
│           ├── mvs-1.npy
│           └── ...
└── ground_truth/     # Ground truth annotations (JSON format)
```

2. Extract motion vectors from videos:
```bash
python utils/extract_motion_vectors.py --video_dir data/videos --output_dir data/motion_vectors
```

## 🔧 Usage

### Basic Usage

```python
from src.mvp_detector import MVPDetector

# Initialize detector
detector = MVPDetector(
    model_id="google/owlv2-large-patch14-ensemble",
    device="cuda"
)

# Process video
results = detector.process_video(
    video_path="path/to/video.mp4",
    motion_vector_dir="path/to/motion_vectors",
    output_dir="path/to/output"
)
```

### Command Line Interface

```bash
# Process single video
python src/main.py --video_path data/videos/video.mp4 --mv_dir data/motion_vectors/video_name --output_dir results/

# Process entire dataset
python src/main.py --video_dir data/videos --mv_root data/motion_vectors --output_root results/
```

### Evaluation

```bash
# Evaluate on dataset
python evaluation/evaluate.py --pred_dir results/ --gt_dir data/ground_truth --output_dir evaluation_results/
```

## 📁 Project Structure

```
MVP/
├── src/                    # Core source code
│   ├── mvp_detector.py    # Main MVP detector class
│   ├── motion_analyzer.py # Motion vector analysis
│   ├── owl_detector.py    # OWL-ViT integration
│   └── main.py            # Command line interface
├── configs/               # Configuration files
│   ├── default.yaml      # Default configuration
│   └── imagenet_vid.yaml # ImageNet VID specific config
├── evaluation/            # Evaluation scripts
│   ├── evaluate.py       # Main evaluation script
│   └── metrics.py        # Evaluation metrics
├── utils/                 # Utility functions
│   ├── extract_motion_vectors.py
│   ├── visualization.py
│   └── data_utils.py
├── examples/              # Usage examples
│   ├── basic_usage.py
│   └── custom_dataset.py
├── mv-extractor/          # Motion vector extraction tool
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎨 Visualization

The framework includes tools for visualizing motion vectors and detection results:

```bash
# Visualize motion vectors
python utils/visualization.py --mv_dir data/motion_vectors/video_name --output_dir visualizations/

# Visualize detection results
python utils/visualization.py --results_dir results/video_name --output_dir visualizations/
```

## 📈 Results

The MVP framework achieves competitive performance on ImageNet VID dataset with significant speedup:

- **mAP@0.2**: 0.747
- **mAP@0.3**: 0.721  
- **mAP@0.5**: 0.609
- **mAP@[0.5:0.95]**: 0.316

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{mvp2025,
  title={MVP: Motion Vector Propagation for Zero-Shot Video Object Detection},
  author={Binhua Huang, Ni Wang, Wendong Yao, Soumyabrata Dev},
  booktitle={2025},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ImageNet VID dataset creators
- OWL-ViT model by Google Research
- Motion vector extraction tool by LukasBommes
- PyTorch and the open-source community
