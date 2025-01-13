# SimpleTransformerLM

A lightweight implementation of a Transformer-based language model in TensorFlow, designed for educational purposes and experimentation. This project includes a custom BPE tokenizer, positional encoding, multi-head attention, and a complete training pipeline.

## Overview

This project implements a small-scale language model using the Transformer architecture, perfect for learning about:
- Transformer architecture implementation
- Custom tokenization with BPE (Byte Pair Encoding)
- Training pipeline development
- Text generation with language models

## Features

- **Custom BPE Tokenizer**: Implements subword tokenization with special token handling
- **Transformer Architecture**:
  - Positional encoding
  - Multi-head attention mechanism
  - Transformer blocks with feed-forward networks
  - Layer normalization and dropout
- **Training Infrastructure**:
  - Memory-efficient data preprocessing
  - Configurable hyperparameters
  - Comprehensive logging
  - TensorBoard integration
  - Checkpoint management
- **Text Generation**:
  - Temperature-based sampling
  - Top-k filtering
  - Dynamic temperature adjustment

## Requirements

```bash
tensorflow>=2.0.0
numpy
psutil
```

## Project Structure

```
├── llm_model.py         # Core model implementation
│   ├── PositionalEncoding
│   ├── MultiHeadAttention
│   ├── TransformerBlock
│   ├── BPETokenizer
│   ├── SimpleLLM
│   └── DataPreprocessor
└── train_llm.py         # Training pipeline
```

## Usage

### 1. Training the Model

```python
from train_llm import main

# Run training with default configuration
main()
```

The training script includes:
- Automatic dataset creation with sample texts
- Model configuration and initialization
- Training loop with callbacks
- Text generation examples
- Memory usage monitoring

### 2. Model Configuration

Default configuration parameters:

```python
config = {
    'batch_size': 2,
    'epochs': 10,
    'max_length': 64,
    'd_model': 256,
    'num_layers': 2,
    'num_heads': 4,
    'dff': 512,
    'learning_rate': 1e-4,
    'vocab_size': 1000,
    'dropout_rate': 0.1
}
```

### 3. Generating Text

```python
from llm_model import SimpleLLM, DataPreprocessor

# Initialize model and preprocessor
model = SimpleLLM(...)
preprocessor = DataPreprocessor()

# Generate text
generated_text = generate_text(
    model,
    preprocessor,
    prompt="The future of technology",
    temperature=0.8,
    max_length=50
)
```

## Training Output

The training process creates a timestamped directory under `./runs/` containing:
- Model checkpoints
- TensorBoard logs
- Generated text samples
- Training metrics

## Implementation Details

### BPE Tokenizer
- Implements subword tokenization
- Handles special tokens (`<pad>`, `<sos>`, `<eos>`, `<unk>`, `<num>`)
- Supports dynamic vocabulary size

### Transformer Architecture
- Configurable number of layers and attention heads
- Implements scaled dot-product attention
- Uses layer normalization and residual connections
- Includes dropout for regularization

### Training Pipeline
- Implements cosine learning rate decay
- Uses sparse categorical crossentropy loss
- Includes early stopping and model checkpointing
- Monitors memory usage throughout training

## Limitations

- Designed for educational purposes and small-scale experiments
- Uses a small sample dataset by default
- Limited vocabulary size (1000 tokens)
- Memory-intensive for large sequences

## Future Improvements

Potential areas for enhancement:
- Implement gradient checkpointing for memory efficiency
- Add support for larger vocabularies
- Implement more sophisticated sampling strategies
- Add data parallelism for multi-GPU training
- Enhance the tokenizer with additional preprocessing options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is inspired by the original Transformer paper: "Attention Is All You Need" (Vaswani et al., 2017).
