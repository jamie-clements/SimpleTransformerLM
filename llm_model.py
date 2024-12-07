# llm_model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import re
from collections import Counter

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
        
        sines = np.sin(angle_rads[:, 0::2])
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
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def call(self, v, k, q, mask=None):
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
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_counts = {}
        self.special_tokens = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3,
            '<num>': 4
        }
        self.vocab = {}
        # Initialize vocab with special tokens
        self.vocab.update(self.special_tokens)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.subword_stats = {}
        
    def train(self, texts):
        # Count words and initialize subwords
        words = []
        for text in texts:
            clean_text = self._preprocess_text(text)
            words.extend(clean_text.split())
        
        # Count word frequencies
        self.word_counts = Counter(words)
        
        # Initialize subword statistics
        for word, count in self.word_counts.items():
            chars = list(word)
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                self.subword_stats[pair] = self.subword_stats.get(pair, 0) + count
        
        # Add most common words and subwords to vocabulary
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab_size_remaining = self.vocab_size - len(self.special_tokens)
        
        # First, add most common full words
        word_vocab_size = vocab_size_remaining // 2
        for word, _ in sorted_words[:word_vocab_size]:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
                self.inverse_vocab[len(self.vocab) - 1] = word
        
        # Then, add most common subwords
        sorted_subwords = sorted(self.subword_stats.items(), key=lambda x: x[1], reverse=True)
        for (char1, char2), _ in sorted_subwords:
            subword = char1 + char2
            if subword not in self.vocab and len(self.vocab) < self.vocab_size:
                self.vocab[subword] = len(self.vocab)
                self.inverse_vocab[len(self.vocab) - 1] = subword
    
    def _preprocess_text(self, text):
        """Preprocess text for tokenization"""
        # Convert to lowercase
        text = text.lower()
        # Replace numbers with <num> token
        text = re.sub(r'\d+', ' <num> ', text)
        # Add spaces around punctuation
        text = re.sub(r'([.,!?()])', r' \1 ', text)
        # Remove special characters
        text = re.sub(r'[^a-z0-9.,!?()\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def encode(self, text, max_length=None):
        """Convert text to token IDs with subword tokenization"""
        # Preprocess the text
        text = self._preprocess_text(text)
        words = text.split()
        
        # Start with special tokens
        tokens = [self.special_tokens['<sos>']]
        
        # Convert words to tokens
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Try subword tokenization
                subwords = self._segment_word(word)
                if subwords:
                    tokens.extend(subwords)
                else:
                    tokens.append(self.special_tokens['<unk>'])
        
        # Add end token
        tokens.append(self.special_tokens['<eos>'])
        
        # Handle max_length
        if max_length is not None:
            if len(tokens) < max_length:
                # Pad sequence
                tokens.extend([self.special_tokens['<pad>']] * (max_length - len(tokens)))
            else:
                # Truncate sequence
                tokens = tokens[:max_length-1] + [self.special_tokens['<eos>']]
        
        return tokens
    
    def _segment_word(self, word):
        """Segment word into subwords"""
        if not word:
            return []
            
        tokens = []
        while word:
            found = False
            # Try to find the longest matching subword
            for i in range(len(word), 0, -1):
                subword = word[:i]
                if subword in self.vocab:
                    tokens.append(self.vocab[subword])
                    word = word[i:]
                    found = True
                    break
            if not found:
                # If no subword is found, take the first character as unknown
                tokens.append(self.special_tokens['<unk>'])
                word = word[1:]
        return tokens
    
    def decode(self, tokens):
        """Convert token IDs back to text"""
        words = []
        current_word = []
        
        for token in tokens:
            if token in self.inverse_vocab:
                word = self.inverse_vocab[token]
                if word in {'<pad>', '<sos>', '<eos>', '<unk>', '<num>'}:
                    if current_word:
                        words.append(''.join(current_word))
                        current_word = []
                    if word == '<num>':
                        words.append('0')
                else:
                    current_word.append(word)
        
        if current_word:
            words.append(''.join(current_word))
        
        return ' '.join(words)

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

class DataPreprocessor:
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.tokenizer = None
        
    def prepare_dataset(self, texts, batch_size=32):
        # Create and train tokenizer if it doesn't exist
        if self.tokenizer is None:
            self.tokenizer = BPETokenizer()
            self.tokenizer.train(texts)
            
        # Tokenize all texts
        encoded_texts = []
        for text in texts:
            encoded = self.tokenizer.encode(text, self.max_length)
            encoded_texts.append(encoded)
            
        # Create training pairs (input, target)
        input_texts = encoded_texts
        target_texts = [text[1:] + [self.tokenizer.special_tokens['<pad>']] 
                       for text in encoded_texts]
        
        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((input_texts, target_texts))
        dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
        
        return dataset