import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices
        sines = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices
        cosines = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add the mask (if provided)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        return output, attention_weights

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        attn_output, _ = self.mha(v=x, k=x, q=x, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class SimpleLLM(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, 
                 maximum_position_encoding, rate=0.1):
        super(SimpleLLM, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)

        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, rate) 
            for _ in range(num_layers)
        ]

        self.dropout = layers.Dropout(rate)
        self.final_layer = layers.Dense(vocab_size)

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        
        # Create masks
        padding_mask = self.create_padding_mask(x)
        look_ahead_mask = self.create_look_ahead_mask(seq_len)
        combined_mask = tf.maximum(padding_mask, look_ahead_mask)

        # Embedding and positional encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training, mask=combined_mask)

        # Final layer
        x = self.final_layer(x)
        return x

# Example usage
def create_model(vocab_size=10000):
    model = SimpleLLM(
        num_layers=6,          # Number of transformer blocks
        d_model=512,          # Embedding dimension
        num_heads=8,          # Number of attention heads
        dff=2048,            # Feed-forward network dimension
        vocab_size=vocab_size,
        maximum_position_encoding=5000
    )
    return model

# Training configuration
def get_training_config():
    return {
        'optimizer': tf.keras.optimizers.Adam(
            learning_rate=CustomSchedule(512),
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        ),
        'loss': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        'metrics': [tf.keras.metrics.SparseCategoricalAccuracy()]
    }

# Learning rate schedule
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# Advanced Tokenization
class BPETokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        
    def train(self, texts):
        # Initialize with characters
        words = []
        for text in texts:
            words.extend(text.split())
        
        # Count character pairs
        char_pairs = {}
        word_pieces = {}
        
        for word in words:
            chars = list(word)
            if len(chars) == 1:
                continue
            
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                char_pairs[pair] = char_pairs.get(pair, 0) + 1
                
        # Merge most frequent pairs until vocab_size is reached
        vocab = list(self.special_tokens.keys())
        for i in range(len(vocab), self.vocab_size):
            if not char_pairs:
                break
                
            best_pair = max(char_pairs.items(), key=lambda x: x[1])[0]
            new_token = ''.join(best_pair)
            vocab.append(new_token)
            
            # Update pairs
            char_pairs = self._update_pairs(char_pairs, best_pair, new_token)
            
        # Create vocabulary
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
    def _update_pairs(self, pairs, pair_to_merge, new_token):
        new_pairs = {}
        for pair, count in pairs.items():
            if pair == pair_to_merge:
                continue
            if pair[0] in pair_to_merge or pair[1] in pair_to_merge:
                continue
            new_pairs[pair] = count
        return new_pairs
        
    def encode(self, text, max_length=None):
        tokens = []
        words = text.split()
        
        tokens.append(self.special_tokens['<sos>'])
        
        for word in words:
            chars = list(word)
            while len(chars) > 1:
                best_pair = None
                best_token = None
                
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    merged = ''.join(pair)
                    if merged in self.vocab:
                        if best_pair is None or len(merged) > len(best_token):
                            best_pair = i
                            best_token = merged
                
                if best_pair is None:
                    break
                    
                chars[best_pair:best_pair + 2] = [best_token]
            
            for piece in chars:
                if piece in self.vocab:
                    tokens.append(self.vocab[piece])
                else:
                    tokens.append(self.special_tokens['<unk>'])
                    
        tokens.append(self.special_tokens['<eos>'])
        
        if max_length is not None:
            if len(tokens) < max_length:
                tokens.extend([self.special_tokens['<pad>']] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]
                
        return tokens
        
    def decode(self, tokens):
        return ' '.join([self.reverse_vocab.get(token, '<unk>') for token in tokens])

import re

# Data Preprocessing
class DataPreprocessor:
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.tokenizer = None
        
    def preprocess_text(self, text):
        # Basic text cleaning
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
    def prepare_dataset(self, texts, batch_size=32):
        # Create and train tokenizer
        if self.tokenizer is None:
            self.tokenizer = BPETokenizer()
            cleaned_texts = [self.preprocess_text(text) for text in texts]
            self.tokenizer.train(cleaned_texts)
            
        # Tokenize all texts
        encoded_texts = []
        for text in texts:
            cleaned = self.preprocess_text(text)
            encoded = self.tokenizer.encode(cleaned, self.max_length)
            encoded_texts.append(encoded)
            
        # Create training pairs (input, target)
        input_texts = encoded_texts
        target_texts = [text[1:] + [self.tokenizer.special_tokens['<pad>']] for text in encoded_texts]
        
        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((input_texts, target_texts))
        dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
        
        return dataset

# Example of how to prepare data
def prepare_data(text, tokenizer, max_length):
    # Tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=max_length,
        padding='post'
    )
    return padded