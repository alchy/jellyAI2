"""
Hlavní program pro trénování a používání word embeddings.
"""

import os
import time
import numpy as np
from typing import List, Tuple
from text_processor import TextProcessor
from simple_nn import MultiLayerWordEmbedding
from config import ModelConfig, TextConfig, TrainingConfig


def load_training_data(processor: TextProcessor) -> List[Tuple[List[int], int]]:
    """Načtení všech trénovacích dat."""
    training_data = []
    for filename in os.listdir(TextConfig.INPUT_DIR):
        if filename.endswith('.txt'):
            with open(os.path.join(TextConfig.INPUT_DIR, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                sequences = processor.get_training_sequences(text)
                training_data.extend(sequences)
    return training_data


def train_model(processor: TextProcessor, model: MultiLayerWordEmbedding, training_data: List[Tuple[List[int], int]]):
    """Trénování modelu."""
    print("Začátek trénování...")
    start_time = time.time()

    num_batches = len(training_data) // ModelConfig.BATCH_SIZE
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(ModelConfig.EPOCHS):
        # Zamíchání dat
        np.random.shuffle(training_data)

        for batch in range(num_batches):
            batch_start = batch * ModelConfig.BATCH_SIZE
            batch_end = batch_start + ModelConfig.BATCH_SIZE
            batch_data = training_data[batch_start:batch_end]

            # Trénování na batchi
            for context_indices, target_idx in batch_data:
                model.backward(context_indices, target_idx, ModelConfig.LEARNING_RATE)

            # Výpis průběhu
            if batch % TrainingConfig.PRINT_EVERY == 0:
                model.print_training_progress(epoch, batch, num_batches)

        # Konec epochy
        model.end_epoch()
        current_loss = model.training_stats['loss'][-1]

        # Early stopping
        if current_loss < best_loss - ModelConfig.MIN_DELTA:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= ModelConfig.PATIENCE:
                print("\nEarly stopping - žádné zlepšení")
                break

    training_time = time.time() - start_time
    print(f"\nTrénování dokončeno za {training_time:.2f} sekund")


def interactive_mode(processor: TextProcessor, model: MultiLayerWordEmbedding):
    """Interaktivní režim pro práci s natrénovaným modelem."""
    print("\nInteraktivní režim - zadávejte slova (nebo 'konec' pro ukončení)")

    while True:
        word = input("\nZadejte slovo: ").strip()
        if word.lower() == 'konec':
            break

        word_idx = processor.word_to_idx(word)  # Změněno na word_to_idx
        if word_idx == 1:  # UNK token
            print(f"Slovo '{word}' není ve slovníku.")
            continue

        # Získání a zobrazení embedding vektoru
        embedding = model.embedding_layer.weights[word_idx]
        print("\nEmbedding vektor (prvních 10 hodnot):")
        print(embedding[:10])

        # Nalezení podobných slov
        similar_words = model.get_similar_words(embedding)
        print("\nPodobná slova:")
        for idx, similarity in similar_words:
            similar_word = processor.idx_to_word(idx)  # Změněno na idx_to_word
            print(f"{similar_word}: {similarity:.4f}")


def main():
    # Kontrola existence vstupního adresáře
    if not os.path.exists(TextConfig.INPUT_DIR):
        os.makedirs(TextConfig.INPUT_DIR)
        print(f"Vytvořen adresář {TextConfig.INPUT_DIR}")
        print("Vložte do něj textové soubory (.txt) a spusťte program znovu.")
        return

    # Inicializace procesoru a načtení dat
    processor = TextProcessor()
    processor.build_vocabulary(TextConfig.INPUT_DIR)

    if processor.unique_words < 3:
        print("Nedostatek dat pro trénování.")
        return

    # Vytvoření modelu
    model = MultiLayerWordEmbedding(
        vocab_size=processor.unique_words,
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        hidden_layers=ModelConfig.HIDDEN_LAYERS,
        activation=ModelConfig.ACTIVATION
    )

    # Načtení trénovacích dat
    training_data = load_training_data(processor)

    if len(training_data) < ModelConfig.BATCH_SIZE:
        print("Nedostatek dat pro trénování.")
        return

    # Trénování
    train_model(processor, model, training_data)

    # Interaktivní režim
    interactive_mode(processor, model)


if __name__ == "__main__":
    main()
