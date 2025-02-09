"""
Modul pro správu slovníku - ukládání a načítání vocabulary.
"""

import os
import json
from typing import List, Dict, Tuple
from datetime import datetime
from config import TextConfig


class DictionaryHandler:
    def __init__(self, dictionary_dir: str = TextConfig.DICTIONARY_DIR):
        """Inicializace handleru pro správu slovníku."""
        self.dictionary_dir = dictionary_dir
        if not os.path.exists(dictionary_dir):
            os.makedirs(dictionary_dir)

    def save_dictionary(self,
                        vocabulary: List[str],
                        word_to_idx: Dict[str, int],
                        word_frequencies: Dict[str, int]) -> str:
        """
        Uloží slovník včetně četností slov a mapování.

        Args:
            vocabulary: Seznam všech slov
            word_to_idx: Mapování slov na indexy
            word_frequencies: Četnosti slov

        Returns:
            str: Cesta k uloženému souboru
        """
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"vocabulary_{timestamp}.json"
        filepath = os.path.join(self.dictionary_dir, filename)

        # Vytvoření kopie nejnovějšího slovníku
        latest_path = os.path.join(self.dictionary_dir, TextConfig.VOCAB_FILENAME)

        dictionary_data = {
            'vocabulary': vocabulary,
            'word_to_idx': word_to_idx,
            'word_frequencies': word_frequencies,
            'metadata': {
                'created_at': timestamp,
                'vocab_size': len(vocabulary),
                'total_words': sum(word_frequencies.values())
            }
        }

        # Uložení slovníku
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dictionary_data, f, ensure_ascii=False, indent=2)

        # Vytvoření kopie jako latest
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(dictionary_data, f, ensure_ascii=False, indent=2)

        return filepath

    def load_dictionary(self, filename: str = None) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
        """
        Načte slovník ze souboru.

        Args:
            filename: Název souboru nebo None pro načtení nejnovějšího slovníku

        Returns:
            Tuple[List[str], Dict[str, int], Dict[str, int]]:
                (vocabulary, word_to_idx, word_frequencies)
        """
        if filename is None:
            filepath = os.path.join(self.dictionary_dir, TextConfig.VOCAB_FILENAME)
        else:
            filepath = os.path.join(self.dictionary_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Slovník {filepath} nebyl nalezen.")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return (
            data['vocabulary'],
            data['word_to_idx'],
            data['word_frequencies']
        )

    def get_dictionary_info(self, filename: str = None) -> Dict:
        """
        Získá informace o slovníku.

        Args:
            filename: Název souboru nebo None pro nejnovější slovník

        Returns:
            Dict: Metadata slovníku
        """
        if filename is None:
            filepath = os.path.join(self.dictionary_dir, TextConfig.VOCAB_FILENAME)
        else:
            filepath = os.path.join(self.dictionary_dir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data['metadata']
