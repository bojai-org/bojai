import torch
import torch.nn as nn


class Model:
    pass


class CLTModelRNN(nn.Module):
    def __init__(self):
        super(CLTModelRNN, self).__init__()

    def initialise(
        self, hidden_size, output_size, batch_size, num_layers, input_size=100
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Output layer to map hidden state to output space
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        rnn_out, hidden = self.rnn(input, hidden)
        output = self.h2o(rnn_out[:, -1, :])

        return output, hidden

    def initHidden(self):
        # Initialize the hidden state (num_layers, batch_size, hidden_size)
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


import string


class CharTokenizer:
    def __init__(self, max_len=100):
        # Define the character set, adding special tokens if necessary
        self.characters = (
            string.ascii_letters + string.digits + string.punctuation + " \n\t"
        )
        self.char_to_index = {char: idx for idx, char in enumerate(self.characters)}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.max_len = max_len  # Maximum length of the sequence

    def build_vocab(self, corpus=None, min_freq=None):
        pass

    def encode(self, text):
        # Encode the text into indices
        encoded = [
            self.char_to_index.get(char, self.char_to_index[" "]) for char in text
        ]

        # Pad or truncate to the max_len
        if len(encoded) < self.max_len:
            encoded.extend([0] * (self.max_len - len(encoded)))  # Padding with 0
        else:
            encoded = encoded[: self.max_len]  # Truncate to max_len

        return torch.tensor(encoded)

    def decode(self, indices):
        # Decode the indices back to the original text
        return "".join([self.index_to_char.get(idx, "") for idx in indices])


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
            self.word_to_index = {
                tok: idx for idx, tok in enumerate(self.special_tokens)
            }
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
        encoded = [
            self.word_to_index.get(word, self.word_to_index["<UNK>"]) for word in tokens
        ]

        # Add <SOS> and <EOS>
        encoded = (
            [self.word_to_index["<SOS>"]] + encoded + [self.word_to_index["<EOS>"]]
        )

        # Padding or truncating
        if len(encoded) < self.max_len:
            encoded.extend(
                [self.word_to_index["<PAD>"]] * (self.max_len - len(encoded))
            )
        else:
            encoded = encoded[: self.max_len]

        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, indices):
        """
        Converts token indices back into text.
        """
        words = [self.index_to_word.get(idx, "<UNK>") for idx in indices]
        return " ".join(
            [word for word in words if word not in ["<PAD>", "<SOS>", "<EOS>"]]
        )
