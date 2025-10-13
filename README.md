# PyTorch GammaNet

A PyTorch implementation of GammaNet - a biologically-inspired recurrent neural network architecture for visual processing with **separate excitatory/inhibitory populations** and horizontal recurrent connections.

**Featured Model**: **VGG16GammaNetV2** - VGG16 backbone + fGRU with E/I populations

## Overview

GammaNet incorporates key biological principles from cortical circuits:

- **Separate E/I Populations**: Explicit excitatory and inhibitory neural populations with Dale's law
- **Horizontal Recurrent Connections**: Within-layer recurrence via fGRU (feedforward Gated Recurrent Unit)
- **Multi-Timestep Processing**: Iterative refinement over 8 recurrent timesteps
- **Biologically-Inspired Constraints**: Symmetric weights, divisive normalization, multiplicative excitation

**Key Features**:
- **VGG16GammaNetV2**: Recommended architecture with E/I populations
- VGG16 backbone with recurrent fGRU layers
- Distribution alignment to match VGG activations
- Support for contour detection, edge detection, and in silico neurophysiology
- Extensive tools for analyzing learned representations and recurrent dynamics

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd pytorch_gammanet

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Optional Dependencies

For in silico experiments (visualization, analysis):
```bash
pip install matplotlib seaborn pandas scipy
```

---

## Getting Started

### Try It Now: Run the Example Script

The easiest way to get started is with our example script:

```bash
# With a trained checkpoint
python example_inference.py \
    --image path/to/your/image.jpg \
    --checkpoint checkpoints/best_model.pt

# Or try with untrained model (random initialization)
python example_inference.py --image path/to/your/image.jpg
```

**Output**: Visualizes input image and detected edges side-by-side, saved to `edge_detection_output.png`

---

### Quick Example: Edge Detection Inference

Here's a complete example using **VGG16GammaNetV2** (recommended) for edge detection:

```python
from gammanet.models import VGG16GammaNetV2
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# 1. Create model with E/I populations (v2 architecture)
config = {
    'layers': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2',
               'conv3_1', 'conv3_2', 'conv3_3', 'pool3'],
    'kernel_size': 3,
    'hidden_channels': 48,
    'num_timesteps': 8,  # 8 recurrent processing steps
    'fgru': {
        'use_separate_ei_states': True,  # Separate E/I populations
        'use_symmetric_conv': True       # Symmetric horizontal connections
    }
}

model = VGG16GammaNetV2(config)
model.eval()

# 2. Load and preprocess image
image = Image.open('example.jpg').convert('RGB')
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)

# 3. Run inference with recurrent processing
with torch.no_grad():
    edge_logits = model(input_tensor)
    edge_prob = torch.sigmoid(edge_logits)

# 4. Visualize results
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title('Input Image')
axes[0].axis('off')
axes[1].imshow(edge_prob[0, 0].cpu(), cmap='gray')
axes[1].set_title('Detected Edges')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('edges_output.png')
print("Saved edge detection result to edges_output.png")
```

**Why VGG16GammaNetV2?**
- **Biologically realistic**: Separate excitatory/inhibitory populations
- **Better for neuroscience**: Can analyze E/I balance and recurrent dynamics
- **Pre-trained VGG16 backbone**: Transfer learning from ImageNet
- **8 timesteps of recurrent refinement**: Iteratively improves edge predictions

---

### Training from Scratch

Train on BSDS500 dataset for contour detection:

```bash
# 1. Ensure data paths are set in config/default.yaml
# 2. Start training
python scripts/train.py --config config/default.yaml

# 3. Monitor training with tensorboard
tensorboard --logdir logs/

# 4. Evaluate best checkpoint
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data-dir /path/to/BSDS500/data \
    --split test
```

**Training details**:
- **Dataset**: BSDS500 (1,000 augmented crops for training)
- **Batch size**: 1 (common for edge detection)
- **Epochs**: 2,000 (small dataset, needs long training)
- **Learning rate**: 0.0003 (fGRU) / 0.00001 (VGG backbone)
- **Time**: ~3-5 days on single GPU

---

### Using Pre-trained Weights

```python
from gammanet.models import VGG16GammaNetV2
import torch

# Load trained model
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cuda')
config = checkpoint['config']

model = VGG16GammaNetV2(config=config['model'])
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
model.cuda()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Validation F1: {checkpoint.get('best_metric', 'N/A')}")
```

---

## Experiments

### In Silico Neurophysiology

Run virtual electrophysiology experiments to compare model responses with primate V1 data:

#### 1. **Orientation Tuning**
```bash
python scripts/run_insilico.py \
    --checkpoint checkpoints/best_model.pt \
    --experiment orientation \
    --output-dir results/orientation
```
Measures orientation selectivity across model units.

#### 2. **Kapadia Collinear Facilitation**
```bash
python scripts/run_insilico.py \
    --checkpoint checkpoints/best_model.pt \
    --experiment kapadia \
    --output-dir results/kapadia
```
Tests center-surround interactions with collinear flankers (Kapadia et al., 1995).

#### 3. **Kinoshita Surround Modulation**
```bash
python scripts/run_insilico.py \
    --checkpoint checkpoints/best_model.pt \
    --experiment kinoshita \
    --output-dir results/kinoshita
```
Analyzes iso-orientation vs. cross-orientation surround suppression (Kinoshita & Gilbert, 2008).

#### 4. **Texture Boundary Detection**
```bash
python scripts/run_insilico.py \
    --checkpoint checkpoints/best_model.pt \
    --experiment texture_boundary \
    --output-dir results/texture_boundary
```
Tests responses to texture boundaries (Trott & Born).

#### 5. **Run All Experiments**
```bash
python scripts/run_insilico.py \
    --checkpoint checkpoints/best_model.pt \
    --experiment all \
    --output-dir results/all_experiments
```

---

### Optogenetic Perturbation Experiments

Simulate optogenetic manipulations to reveal recurrent circuit structure:

#### **Orientation-Based Multi-Orientation Optimization**

This experiment perturbs the feedforward input and learns what initial E/I states in the surround would rescue orientation decoding. By optimizing across multiple orientations and averaging, it reveals the **orientation-invariant recurrent receptive field structure**.

```bash
python scripts/test_orientation_opto.py
```

**What it does**:
1. **Parameter Sweep**: Finds optimal spatial frequency and contrast for CRF orientation decoding
2. **Decoder Training**: Trains orientation decoder on CRF-only gratings (5px diameter)
3. **Approach A**: Measures actual recurrent E/I flow using learned weights
4. **Approach B**: Multi-orientation optimization
   - Perturbs feedforward input at CRF center (50% increase)
   - Optimizes initial E/I states in eCRF surround (37px diameter)
   - Tests 6 orientations (0°, 30°, 60°, 90°, 120°, 150°)
   - Rotates each to 0° reference and averages
   - **Result**: Canonical orientation-invariant RF structure

**Output**:
- `results/orientation_opto/parameter_sweep.png` - R² heatmap for SF/contrast selection
- `results/orientation_opto/stimuli.png` - CRF vs eCRF grating visualizations
- `results/orientation_opto/recurrent_flow_analysis.png` - E/I flow to perturbed center
- `results/orientation_opto/multi_orientation_optimization_results.png` - Averaged RF structure
- `results/orientation_opto/summary.txt` - Full results summary

**Key Findings**:
- Reveals fundamental recurrent connectivity pattern
- Shows how surround rescues perturbed center
- Orientation-invariant structure indicates canonical RF topology

---

## Project Structure

```
pytorch_gammanet/
├── gammanet/                    # Core library
│   ├── models/                  # Network architectures
│   │   ├── vgg16_gammanet.py   # VGG16 + fGRU (original)
│   │   ├── vgg16_gammanet_v2.py # VGG16 + fGRU (E/I version)
│   │   └── components/          # fGRU, alignment, etc.
│   ├── data/                    # Dataset loaders
│   ├── training/                # Training utilities
│   ├── utils/                   # Metrics, losses, etc.
│   └── analysis/                # Analysis tools
│       └── optogenetic_perturbation.py  # Opto simulation
│
├── experiments/                 # In silico experiments
│   └── in_silico/
│       ├── stimuli.py          # Stimulus generation (gratings, Gabors, etc.)
│       ├── extract.py          # Response extraction
│       ├── analysis.py         # Tuning analysis
│       ├── neural_comparison.py # Model-neural alignment
│       └── visualize.py        # Plotting utilities
│
├── scripts/                     # Executable scripts
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── run_insilico.py         # In silico experiments
│   └── test_orientation_opto.py # Optogenetic simulations
│
├── config/                      # Configuration files
│   ├── default.yaml            # Default config
│   └── ablations/              # Ablation study configs
│
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
├── example_inference.py        # Quick start inference example
└── README.md                   # This file
```

---

## Configuration

### Model Architecture

Edit `config/default.yaml` to customize:

**fGRU Hyperparameters**:
```yaml
model:
  timesteps: 8                   # Number of recurrent timesteps
  model_version: v2              # v1 (original) or v2 (E/I populations)

  fgru:
    use_separate_ei_states: true # Separate E/I populations (Dale's law)
    use_symmetric_conv: true     # Symmetric horizontal connections
    multiplicative_excitation: true
    use_attention: null          # null, 'se', 'gala'
```

**Training**:
```yaml
training:
  batch_size: 1
  num_epochs: 2000
  learning_rate: 0.0003          # fGRU learning rate
  vgg_learning_rate: 0.00001     # VGG backbone learning rate (10x lower)
  loss: bi_bce_hed               # Loss function
  mixed_precision: false         # Use AMP for faster training
```

**Data Paths**:
```yaml
data:
  dataset: bsds500
  train_path: /path/to/BSDS500_crops/data
  val_path: /path/to/BSDS500_crops/data
  test_path: /path/to/BSDS500/data
```

---

## Advanced Usage

### Custom In Silico Experiments

```python
from experiments.in_silico import (
    OrientedGratingStimuli,
    ResponseExtractor,
    OrientationTuningAnalyzer
)

# Generate stimuli
stim_gen = OrientedGratingStimuli(size=(256, 256))
stimuli = stim_gen.generate_stimulus_set(
    orientations=[0, 30, 60, 90, 120, 150],
    spatial_frequencies=[20.0, 30.0, 40.0],
    contrasts=[0.5, 1.0],
    stimulus_diameter=37  # eCRF size
)

# Extract responses
extractor = ResponseExtractor(model, device='cuda')
responses = extractor.extract_responses(
    stimuli,
    layer_name='fgru_0',
    timesteps=8
)

# Analyze orientation tuning
analyzer = OrientationTuningAnalyzer()
tuning_data = analyzer.compute_tuning_curves(responses, stimuli)
```

### Custom Optogenetic Experiments

```python
from gammanet.analysis import OptogeneticPerturbation

perturber = OptogeneticPerturbation(model, device='cuda', patch_size=5)

# Train orientation decoder
perturber.train_decoder(
    crf_grating_stimuli,
    layer_name='fgru_0',
    decoder_location=(128, 128),
    num_epochs=100
)

# Multi-orientation optimization
result = perturber.optimize_circuit_response_multi_orientation(
    grating_stimuli,
    layer_name='fgru_0',
    perturb_location=(128, 128),
    perturb_factor=1.5,  # 50% increase
    num_steps=50,
    ecrf_radius=18
)

# Visualize
from gammanet.analysis import visualize_multi_orientation_optimization
visualize_multi_orientation_optimization(
    result,
    save_path='optimization_results.png',
    show_individual=True
)
```

---

## Model Variants

### VGG16GammaNet (v1)
- Original implementation with single hidden state per layer
- Standard recurrent processing

### VGG16GammaNetV2 (v2) - Recommended
- **Separate E/I populations** with Dale's law
- More biologically realistic
- Better for neurophysiology experiments

### Usage:
```python
# v1 (original)
from gammanet.models import VGG16GammaNet
model = VGG16GammaNet(config)

# v2 (E/I populations)
from gammanet.models import VGG16GammaNetV2
model = VGG16GammaNetV2(config)
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{gammanet2024,
  title={GammaNet: Biologically-Inspired Recurrent Neural Networks with Separate Excitatory/Inhibitory Populations},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

Based on the original TensorFlow implementation and neurophysiology insights from:
- Kapadia et al. (1995) - Collinear facilitation
- Kinoshita & Gilbert (2008) - Surround modulation
- Trott & Born - Texture boundary detection

---

## License

MIT License - See LICENSE file for details

---

## Troubleshooting

**CUDA Out of Memory**

Reduce batch size or number of timesteps:
```yaml
training:
  batch_size: 1
model:
  timesteps: 4
```

**Slow Training**

Enable mixed precision:
```yaml
training:
  mixed_precision: true
```

**Poor Convergence**

- Check learning rates (VGG backbone should be 10x lower than fGRU)
- Verify data paths are correct
- Ensure BSDS500 images are properly normalized

---

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

## Acknowledgments

This work builds on insights from computational neuroscience and cortical circuit models. Special thanks to the original GammaNet authors and the neurophysiology community for foundational experiments that guide this research.
