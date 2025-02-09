"""
Konfigurační soubor pro nastavení parametrů zpracování textu a neuronové sítě.
"""

import os
from datetime import datetime

class ModelConfig:
    # Parametry architektury sítě
    EMBEDDING_DIM = 100
    HIDDEN_LAYERS = [64, 32]
    ACTIVATION = 'relu'

    # Parametry trénování
    LEARNING_RATE = 0.01
    EPOCHS = 10
    BATCH_SIZE = 32

    # Early stopping
    PATIENCE = 3
    MIN_DELTA = 0.001

class TextConfig:
    # Parametry pro zpracování textu
    MIN_WORD_FREQUENCY = 2
    MAX_SEQUENCE_LENGTH = 50
    CONTEXT_WINDOW_SIZE = 2

    # Velikost chunku pro postupné trénování (počet vět)
    CHUNK_SIZE = 1000

    # Speciální tokeny
    PAD_TOKEN = '<PAD>'  # Token pro padding
    UNK_TOKEN = '<UNK>'  # Token pro neznámá slova
    BOS_TOKEN = '<BOS>'  # Token pro začátek věty
    EOS_TOKEN = '<EOS>'  # Token pro konec věty

    # Cesty
    INPUT_DIR = 'input_text'
    MODELS_DIR = 'saved_models'
    DICTIONARY_DIR = 'dictionary'
    VOCAB_FILENAME = 'vocabulary.json'

class ModelManager:
    """Správa ukládání a načítání modelů."""

    @staticmethod
    def get_model_filename(loss: float, epoch: int, timestamp: str = None) -> str:
        """Vytvoří název souboru modelu obsahující informace o jeho kvalitě."""
        if timestamp is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"model_loss_{loss:.4f}_epoch_{epoch}_{timestamp}.npz"

    @staticmethod
    def parse_model_info(filename: str) -> dict:
        """Získá informace o modelu z jeho názvu."""
        try:
            parts = filename.replace('.npz', '').split('_')
            return {
                'loss': float(parts[2]),
                'epoch': int(parts[4]),
                'timestamp': '_'.join(parts[5:])
            }
        except:
            return None

    @staticmethod
    def get_best_model() -> str:
        """Najde nejlepší model podle loss hodnoty."""
        if not os.path.exists(TextConfig.MODELS_DIR):
            return None

        models = []
        for filename in os.listdir(TextConfig.MODELS_DIR):
            if filename.endswith('.npz'):
                info = ModelManager.parse_model_info(filename)
                if info:
                    models.append((filename, info['loss']))

        if not models:
            return None

        return min(models, key=lambda x: x[1])[0]

class TrainingConfig:
    # Parametry pro monitoring trénování
    PRINT_EVERY = 1000
    SAVE_EVERY = 5000

    # Logování
    LOG_FILE = 'training.log'
    VERBOSE = True

    # Parametry trénování
    BATCH_SIZE = ModelConfig.BATCH_SIZE
    CONTEXT_WINDOW_SIZE = TextConfig.CONTEXT_WINDOW_SIZE
