# train_llm.py
import tensorflow as tf
import numpy as np
import os
import logging
import traceback
import psutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our LLM code
from llm_model import SimpleLLM, DataPreprocessor

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def create_sample_dataset():
    """Create a small dataset for training example"""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "I love learning about artificial intelligence and machine learning.",
        "Python is a versatile programming language used in data science.",
        "Neural networks are inspired by biological brains.",
        "Deep learning has revolutionized natural language processing.",
        "Transformers have become the standard for language models.",
        "Machine learning models learn patterns from data.",
        "Artificial intelligence is changing the world rapidly.",
        "Data science combines statistics and programming.",
        "Computer vision helps machines understand images."
    ]

def test_model_forward_pass(model, test_input):
    """Test the model's forward pass step by step"""
    logger.info("Testing model forward pass step by step...")
    try:
        # Test embedding layer
        embedded = model.embedding(test_input)
        logger.info(f"Embedding shape: {embedded.shape}")
        
        # Test positional encoding
        pos_encoded = model.pos_encoding(embedded)
        logger.info(f"Positional encoding shape: {pos_encoded.shape}")
        
        # Test transformer blocks
        block_input = pos_encoded
        for i, block in enumerate(model.transformer_blocks):
            block_output = block(
                block_input, 
                training=False,
                mask=None
            )
            logger.info(f"Transformer block {i} output shape: {block_output.shape}")
            block_input = block_output
            
        # Test final layer
        final_output = model.final_layer(block_input)
        logger.info(f"Final output shape: {final_output.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Forward pass test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def generate_text(model, preprocessor, prompt, max_length=30, temperature=1.0):
    """Generate text from a prompt with temperature sampling"""
    logger.info(f"Generating text for prompt: '{prompt}'")
    try:
        input_ids = preprocessor.tokenizer.encode(prompt)
        input_ids = tf.cast(tf.expand_dims(input_ids, 0), tf.int32)  # Cast to int32
        
        for _ in range(max_length):
            predictions = model(input_ids, training=False)
            predictions = predictions[:, -1, :] / temperature  # Remove the extra dimension
            predictions = tf.reshape(predictions, (1, -1))     # Reshape to (batch_size, vocab_size)
            
            # Generate and cast to int32
            predicted_id = tf.cast(
                tf.random.categorical(predictions, num_samples=1)[0], 
                tf.int32
            )
            
            # Ensure both tensors are int32 before concatenation
            input_ids = tf.concat([
                input_ids,
                tf.reshape(predicted_id, (1, 1))  # Reshape to match input_ids dimensions
            ], axis=1)
            
            if predicted_id == preprocessor.tokenizer.special_tokens['<eos>']:
                break
        
        return preprocessor.tokenizer.decode(input_ids[0].numpy())
    except Exception as e:
        logger.error(f"Error in text generation: {str(e)}")
        return f"Error generating text: {str(e)}"

def main():
    # Create output directory for logs and checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./runs/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting LLM training script. Output directory: {output_dir}")
    logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    # Model configuration
    config = {
        'batch_size': 2,
        'epochs': 5,
        'max_length': 64,
        'd_model': 256,
        'num_layers': 2,
        'num_heads': 4,
        'dff': 512,
        'learning_rate': 1e-4
    }
    
    logger.info("Model configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Prepare Dataset
    logger.info("Preparing dataset...")
    try:
        training_texts = create_sample_dataset()
        preprocessor = DataPreprocessor(max_length=config['max_length'])
        dataset = preprocessor.prepare_dataset(
            texts=training_texts,
            batch_size=config['batch_size']
        )
        logger.info("Dataset preparation completed successfully")
        logger.info(f"Memory usage after dataset preparation: {get_memory_usage():.2f} MB")
    except Exception as e:
        logger.error(f"Error during dataset preparation: {str(e)}")
        logger.error(traceback.format_exc())
        return

    # Create Model
    logger.info("Creating model...")
    try:
        model = SimpleLLM(
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            dff=config['dff'],
            vocab_size=preprocessor.tokenizer.vocab_size,
            maximum_position_encoding=config['max_length']
        )
        logger.info("Model created successfully")
        logger.info(f"Memory usage after model creation: {get_memory_usage():.2f} MB")
    except Exception as e:
        logger.error(f"Error during model creation: {str(e)}")
        logger.error(traceback.format_exc())
        return

    # Compile Model
    logger.info("Compiling model...")
    try:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        logger.info("Model compiled successfully")
    except Exception as e:
        logger.error(f"Error during model compilation: {str(e)}")
        logger.error(traceback.format_exc())
        return

    # Setup callbacks
    def generate_sample_callback(epoch, logs):
        logger.info(f"\nGenerating sample for epoch {epoch}:")
        generated = generate_text(
            model, 
            preprocessor, 
            "The quick brown",
            temperature=0.7
        )
        logger.info(f"Sample: {generated}\n")
        logger.info(f"Memory usage: {get_memory_usage():.2f} MB")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'model_checkpoint_{epoch}.weights.h5'),
            save_weights_only=True,
            save_best_only=True,
            monitor='loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            update_freq='epoch'
        ),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=generate_sample_callback
        )
    ]

    # Train Model
    logger.info("Starting training...")
    try:
        # Verify dataset structure
        for batch in dataset.take(1):
            logger.info(f"Input batch shape: {batch[0].shape}")
            logger.info(f"Target batch shape: {batch[1].shape}")
        
        # Test forward pass
        logger.info("Testing forward pass...")
        test_input = next(iter(dataset))[0]
        
        if not test_model_forward_pass(model, test_input):
            logger.error("Model forward pass test failed. Aborting training.")
            return
            
        logger.info("Forward pass successful!")
        logger.info(f"Memory usage before training: {get_memory_usage():.2f} MB")
        
        history = model.fit(
            dataset,
            epochs=config['epochs'],
            callbacks=callbacks
        )
        logger.info("Training completed successfully!")
        
        # Save final model weights
        final_weights_path = os.path.join(output_dir, 'final.weights.h5')
        model.save_weights(final_weights_path)
        logger.info(f"Final weights saved to: {final_weights_path}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error(traceback.format_exc())
        return

    # Generate example texts
    logger.info("\nGenerating example texts...")
    prompts = [
        "The quick",
        "I love",
        "Python is"
    ]
    
    for prompt in prompts:
        generated_text = generate_text(model, preprocessor, prompt, temperature=0.7)
        logger.info(f"\nPrompt: {prompt}")
        logger.info(f"Generated: {generated_text}")

    logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")
    logger.info("Script completed successfully!")

if __name__ == "__main__":
    main()