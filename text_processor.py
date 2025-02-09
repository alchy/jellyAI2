"""
Zpracování textových dat pro neuronovou síť.
"""

import os
import re
import unicodedata
from typing import List, Dict, Set
from config import TextConfig as cfg

class TextProcessor:
    def __init__(self):
        """Inicializace procesoru pro zpracování textu."""
        # Základní slovníky
        self.vocabulary = [cfg.PAD_TOKEN, cfg.UNK_TOKEN]
        self._word_to_idx = {cfg.PAD_TOKEN: 0, cfg.UNK_TOKEN: 1}  # Přejmenováno
        self.word_frequencies = {}

        # Statistiky
        self.total_words = 0
        self.unique_words = 0

    def remove_diacritics(self, text: str) -> str:
        """Odstranění diakritiky z textu."""
        nfkd_form = unicodedata.normalize('NFKD', text)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def clean_text(self, text: str) -> str:
        """
        Základní čištění textu.
        - převod na malá písmena
        - odstranění diakritiky
        - odstranění speciálních znaků
        """
        text = text.lower()
        text = self.remove_diacritics(text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def build_vocabulary(self, input_dir: str):
        """Vytvoření slovníku ze všech textových souborů v adresáři."""
        print(f"Načítání textů z: {input_dir}")

        # Počítání četnosti slov
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file:
                    text = self.clean_text(file.read())
                    words = text.split()
                    self.total_words += len(words)

                    for word in words:
                        self.word_frequencies[word] = self.word_frequencies.get(word, 0) + 1

        # Vytvoření slovníku (pouze dostatečně častá slova)
        for word, freq in self.word_frequencies.items():
            if freq >= cfg.MIN_WORD_FREQUENCY and word not in self._word_to_idx:
                self._word_to_idx[word] = len(self.vocabulary)
                self.vocabulary.append(word)

        self.unique_words = len(self.vocabulary)
        print(f"Celkem slov: {self.total_words}")
        print(f"Unikátních slov ve slovníku: {self.unique_words}")

    def get_training_sequences(self, text: str) -> List[List[int]]:
        """Vytvoření trénovacích sekvencí z textu."""
        words = self.clean_text(text).split()
        sequences = []

        for i in range(len(words)):
            if i >= cfg.CONTEXT_WINDOW_SIZE and i < len(words) - cfg.CONTEXT_WINDOW_SIZE:
                context = (words[i-cfg.CONTEXT_WINDOW_SIZE:i] +
                          words[i+1:i+cfg.CONTEXT_WINDOW_SIZE+1])
                target = words[i]

                context_indices = [self.word_to_idx(w) for w in context]
                target_idx = self.word_to_idx(target)

                if len(context_indices) == 2 * cfg.CONTEXT_WINDOW_SIZE:
                    sequences.append((context_indices, target_idx))

        return sequences

    def word_to_idx(self, word: str) -> int:
        """Převod slova na index."""
        return self._word_to_idx.get(self.clean_text(word), 1)

    def idx_to_word(self, idx: int) -> str:
        """Převod indexu na slovo."""
        if 0 <= idx < len(self.vocabulary):
            return self.vocabulary[idx]
        return cfg.UNK_TOKEN