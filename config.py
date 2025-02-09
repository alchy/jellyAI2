"""
Konfigurační soubor pro nastavení parametrů zpracování textu a neuronové sítě.
"""

class ModelConfig:
    # Parametry architektury sítě
    EMBEDDING_DIM = 100
    HIDDEN_LAYERS = [64, 32]  # Velikosti skrytých vrstev
    ACTIVATION = 'relu'  # Možnosti: 'relu', 'tanh'

    # Parametry trénování
    LEARNING_RATE = 0.01
    EPOCHS = 10
    BATCH_SIZE = 32

    # Early stopping
    PATIENCE = 3  # Počet epoch bez zlepšení
    MIN_DELTA = 0.001  # Minimální změna pro považování za zlepšení

class TextConfig:
    # Parametry pro zpracování textu
    MIN_WORD_FREQUENCY = 2  # Minimální četnost slova pro zahrnutí do slovníku
    MAX_SEQUENCE_LENGTH = 50  # Maximální délka sekvence pro trénování
    CONTEXT_WINDOW_SIZE = 2  # Počet slov na každé straně cílového slova

    # Speciální tokeny
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'

    # Cesty
    INPUT_DIR = 'input_text'  # Adresář se vstupními texty
    MODEL_SAVE_PATH = 'saved_model'  # Adresář pro ukládání modelu

class TrainingConfig:
    # Parametry pro monitoring trénování
    PRINT_EVERY = 1000  # Četnost výpisů během trénování
    SAVE_EVERY = 5000  # Četnost ukládání modelu

    # Logování
    LOG_FILE = 'training.log'
    VERBOSE = True  # Detailní výpisy

    # Parametry trénování
    BATCH_SIZE = ModelConfig.BATCH_SIZE  # Použití stejné velikosti batch jako v ModelConfig
    CONTEXT_WINDOW_SIZE = TextConfig.CONTEXT_WINDOW_SIZE  # Použití stejné velikosti okna jako v TextConfig