# Cat vs Dog Classifier with PyTorch Lightning

Implementation from MLOps Course - Session 04

A production-ready MLOps implementation of a cat vs dog image classifier using PyTorch Lightning, featuring comprehensive logging, monitoring, and a complete ML pipeline.

## 🌟 Features

- 🚀 MobileNetV2-based transfer learning
- 📊 Comprehensive logging with Loguru and TensorBoard
- 🎯 Rich progress tracking and visualization
- 🔄 Complete ML pipeline (train/validate/test/infer)
- 🛠 Production-ready project structure
- 📈 Performance visualization with confidence scores
- 🎨 Beautiful prediction visualizations

## 🏗 Project Structure

```
mlops-practice/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── datamodule.py      # Dataset handling
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py      # MobileNetV2 model
│   ├── utils.py              # Logging utilities
│   ├── train.py             # Training pipeline
│   ├── infer.py             # Inference with viz
│   └── download_samples.py   # Sample images
├── configs/
│   └── config.yaml          # Training config
├── samples/                 # Test images
├── predictions/             # Model outputs
├── logs/                    # Training logs
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- uv package manager
- CUDA-capable GPU (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlops-practice.git
cd mlops-practice
```

2. Create virtual environment using uv:
```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. Install dependencies with uv:
```bash
uv pip install -r requirements.txt
# or
uv pip install -e .
```

### Training Pipeline

1. Configure training in `configs/config.yaml`:
```yaml
model:
  lr: 1e-3
data:
  batch_size: 32
  num_workers: 2
trainer:
  max_epochs: 5
  accelerator: "auto"
```

2. Start training:
```bash
python src/train.py
```

Monitor training:
```bash
tensorboard --logdir logs/catdog_classification
```

### Inference Pipeline

1. Download sample images:
```bash
python src/download_samples.py
```

2. Run inference:
```bash
python src/infer.py \
    -input_folder src/samples \
    -output_folder predictions \
    -ckpt "$(cat best_model_path.txt)"
```

## 🔧 Model Architecture

- Base: MobileNetV2 (pretrained)
- Input: 96x96 RGB images
- Output: Binary classification (Cat/Dog)
- Training:
  - Optimizer: Adam (lr=1e-3)
  - Loss: Cross Entropy
  - Metrics: Accuracy
  - Batch size: 4 (CPU) / 32 (GPU)
  - Gradient accumulation: 8 steps

## 📊 Logging & Monitoring

### Training Logs
- `logs/catdog_classification/`: TensorBoard metrics
- `logs/app.log`: Execution logs

### Inference Outputs
- Individual predictions with:
  - Confidence scores
  - Color-coded borders (green: >0.8, orange: >0.5, red: <0.5)
  - Confidence bars
- Summary grid of all predictions
- `predictions_summary.txt`

## 🔍 Code Components

### DataModule
- Downloads cat/dog dataset
- Implements data transforms
- Handles data loading
```python
data_module = CatDogImageDataModule(
    batch_size=4,
    num_workers=0
)
```

### Model
- MobileNetV2 with transfer learning
- PyTorch Lightning implementation
```python
model = CatDogClassifier(lr=1e-3)
```

### Training
- PyTorch Lightning Trainer
- Rich progress bars
- Model checkpointing
```python
trainer = L.Trainer(
    max_epochs=5,
    accelerator="cpu",
    precision="32"
)
```

## 🐛 Troubleshooting

### Common Issues

1. Memory Errors
```bash
# Reduce batch size in config.yaml
data:
  batch_size: 4
```

2. CUDA Issues
```bash
# Switch to CPU in config.yaml
trainer:
  accelerator: "cpu"
```

3. Path Issues
```bash
# Create required directories
mkdir -p predictions logs/catdog_classification/checkpoints
```

## 📈 Performance

- Training accuracy: ~95%
- Validation accuracy: ~93%
- Test accuracy: ~92%

## 🛠 Development

Using uv for dependency management:
```bash
# Add new dependency
uv pip install package_name

# Update dependencies
uv pip freeze > requirements.txt

# Sync environment
uv pip sync requirements.txt
```

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## 📚 References

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [uv Package Manager](https://github.com/astral-sh/uv)
