import torch
import torch.nn as nn


class Model():
    pass

class CLIModelCNN(nn.Module):
    def __init__(self):
        super(CLIModelCNN, self).__init__()

    def initialise(self, input_size : tuple, output_size : int):
        in_ch, h,w = input_size
        self.cnn1 = nn.Conv2d(in_channels = in_ch, out_channels = 32, kernel_size = 3)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        self.flatten = nn.Flatten()
        self.output = nn.Linear(968256, output_size)
        print(input_size)

        
        
    def forward(self, input):
        input = self.relu(self.cnn1(input))
        input = self.pooling(input)
        input = self.relu(self.cnn2(input))
        input = self.pooling(input)
        flat = self.flatten(input)

        print(flat.shape)
        output = self.output(flat)

        return output




import string

class CharTokenizer:
    def __init__(self, max_len=100):
        # Define the character set, adding special tokens if necessary
        self.characters = string.ascii_letters + string.digits + string.punctuation + " \n\t"
        self.char_to_index = {char: idx for idx, char in enumerate(self.characters)}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.max_len = max_len  # Maximum length of the sequence
    
    def encode(self, text):
        # Encode the text into indices
        encoded = [self.char_to_index.get(char, self.char_to_index[' ']) for char in text]
        
        # Pad or truncate to the max_len
        if len(encoded) < self.max_len:
            encoded.extend([0] * (self.max_len - len(encoded)))  # Padding with 0
        else:
            encoded = encoded[:self.max_len]  # Truncate to max_len
        
        return torch.tensor(encoded)
    
    def decode(self, indices):
        # Decode the indices back to the original text
        return ''.join([self.index_to_char.get(idx, '') for idx in indices])


import torch
import string
from collections import Counter

class VocabTokenizer:
    def __init__(self, corpus=None, max_len=100, min_freq=1):
        """
        Initializes the tokenizer with a vocabulary built from the given corpus.
        :param corpus: List of texts to build vocabulary from (optional).
        :param max_len: Maximum sequence length.
        :param min_freq: Minimum frequency for a token to be included in the vocabulary.
        """
        self.max_len = max_len
        self.special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        
        if corpus:
            self.build_vocab(corpus, min_freq)
        else:
            self.word_to_index = {tok: idx for idx, tok in enumerate(self.special_tokens)}
            self.index_to_word = {idx: tok for tok, idx in self.word_to_index.items()}

    def build_vocab(self, corpus, min_freq):
        """
        Builds a vocabulary from a corpus, filtering words with frequency < min_freq.
        """
        word_counts = Counter()
        for text in corpus:
            word_counts.update(text.split())

        vocab = [word for word, count in word_counts.items() if count >= min_freq]
        vocab = self.special_tokens + vocab  # Add special tokens at the start

        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}

    def encode(self, text):
        """
        Tokenizes text into indices, adds <SOS> and <EOS>, and pads/truncates to max_len.
        """
        tokens = text.split()
        encoded = [self.word_to_index.get(word, self.word_to_index["<UNK>"]) for word in tokens]
        
        # Add <SOS> and <EOS>
        encoded = [self.word_to_index["<SOS>"]] + encoded + [self.word_to_index["<EOS>"]]

        # Padding or truncating
        if len(encoded) < self.max_len:
            encoded.extend([self.word_to_index["<PAD>"]] * (self.max_len - len(encoded)))
        else:
            encoded = encoded[:self.max_len]

        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, indices):
        """
        Converts token indices back into text.
        """
        words = [self.index_to_word.get(idx, "<UNK>") for idx in indices]
        return ' '.join([word for word in words if word not in ["<PAD>", "<SOS>", "<EOS>"]])