"""
Hlavní program pro trénování a používání word embeddings s podporou
postupného trénování a správy modelů.
"""

import os
import time
from typing import List, Tuple
import numpy as np
from datetime import datetime

from text_processor import TextProcessor
from simple_nn import MultiLayerWordEmbedding
from model_handler import ModelHandler
from config import ModelConfig, TextConfig, TrainingConfig

def load_training_data(processor: TextProcessor) -> List[Tuple[List[int], int]]:
    """
    Načtení všech trénovacích dat.

    Args:
        processor: Instance TextProcessor pro zpracování textu

    Returns:
        List[Tuple[List[int], int]]: Seznam dvojic (kontext, cílový index)
    """
    training_data = []
    print("\nNačítání trénovacích dat...")

    total_files = sum(1 for f in os.listdir(TextConfig.INPUT_DIR) if f.endswith('.txt'))
    processed_files = 0

    for filename in os.listdir(TextConfig.INPUT_DIR):
        if filename.endswith('.txt'):
            processed_files += 1
            print(f"\rZpracování souborů: {processed_files}/{total_files}", end="")

            with open(os.path.join(TextConfig.INPUT_DIR, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                sequences = processor.get_training_sequences(text)
                training_data.extend(sequences)

    print(f"\nCelkem načteno {len(training_data)} trénovacích sekvencí")
    return training_data

def train_model_incrementally(processor: TextProcessor,
                            model: MultiLayerWordEmbedding,
                            training_data: List[Tuple[List[int], int]],
                            model_handler: ModelHandler):
    """
    Postupné trénování modelu s ukládáním nejlepších vah.

    Args:
        processor: Instance TextProcessor
        model: Instance neuronové sítě
        training_data: Trénovací data
        model_handler: Instance pro správu modelů
    """
    print("\nZačátek postupného trénování...")
    start_time = time.time()

    # Rozdělení dat na chunky
    chunks = [training_data[i:i + TextConfig.CHUNK_SIZE]
             for i in range(0, len(training_data), TextConfig.CHUNK_SIZE)]

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(ModelConfig.EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpocha {epoch + 1}/{ModelConfig.EPOCHS}")

        total_chunks = len(chunks)
        for chunk_idx, chunk in enumerate(chunks):
            print(f"\nZpracování chunk {chunk_idx + 1}/{total_chunks}")
            chunk_start_time = time.time()

            # Trénování na chunku
            total_batches = len(chunk) // ModelConfig.BATCH_SIZE
            for batch_idx in range(0, len(chunk), ModelConfig.BATCH_SIZE):
                batch = chunk[batch_idx:batch_idx + ModelConfig.BATCH_SIZE]

                # Trénování na batchi
                batch_loss = 0
                for context_indices, target_idx in batch:
                    model.backward(context_indices, target_idx, ModelConfig.LEARNING_RATE)
                    batch_loss += model.epoch_stats['loss']

                # Výpis průběhu
                if batch_idx % TrainingConfig.PRINT_EVERY == 0:
                    avg_loss = batch_loss / len(batch)
                    progress = (batch_idx + 1) / len(chunk) * 100
                    elapsed = time.time() - chunk_start_time
                    print(f"\rProgress: {progress:.1f}% | Loss: {avg_loss:.4f} | "
                          f"Elapsed: {elapsed:.1f}s", end="")

        # Konec epochy
        model.end_epoch()
        current_loss = model.training_stats['loss'][-1]
        epoch_time = time.time() - epoch_start_time

        print(f"\nEpocha {epoch + 1} dokončena za {epoch_time:.1f}s | "
              f"Loss: {current_loss:.4f}")

        # Ukládání modelu při zlepšení
        if current_loss < best_loss - ModelConfig.MIN_DELTA:
            best_loss = current_loss
            patience_counter = 0

            # Uložení modelu
            filename = model_handler.save_model(
                model,
                current_loss,
                epoch + 1,
                processor.vocabulary
            )
            print(f"Uložen nový nejlepší model: {filename}")
        else:
            patience_counter += 1
            print(f"Není zlepšení. Patience: {patience_counter}/{ModelConfig.PATIENCE}")
            if patience_counter >= ModelConfig.PATIENCE:
                print("Early stopping - žádné zlepšení")
                break

    total_time = time.time() - start_time
    print(f"\nTrénování dokončeno za {total_time:.1f} sekund")
    print(f"Nejlepší dosažená loss: {best_loss:.4f}")

def interactive_mode(processor: TextProcessor, model: MultiLayerWordEmbedding):
    """
    Interaktivní režim pro práci s natrénovaným modelem.

    Args:
        processor: Instance TextProcessor
        model: Natrénovaný model
    """
    print("\nInteraktivní režim - zadávejte slova (nebo 'konec' pro ukončení)")
    print("Pro každé slovo zobrazím jeho embedding vektor a nejpodobnější slova.")

    while True:
        word = input("\nZadejte slovo: ").strip()
        if word.lower() == 'konec':
            break

        word_idx = processor.word_to_idx(word)
        if word_idx == 1:  # UNK token
            print(f"Slovo '{word}' není ve slovníku.")
            continue

        # Získání a zobrazení embedding vektoru
        embedding = model.embedding_layer.weights[word_idx]
        print("\nEmbedding vektor (prvních 10 hodnot):")
        print(embedding[:10])
        print(f"Délka vektoru: {len(embedding)}")

        # Nalezení podobných slov
        similar_words = model.get_similar_words(embedding)
        print("\nNejpodobnější slova:")
        for idx, similarity in similar_words:
            similar_word = processor.idx_to_word(idx)
            print(f"{similar_word}: {similarity:.4f}")


def main():
    """Hlavní funkce programu."""
    print("Word Embeddings Trainer")
    print("----------------------")

    # Kontrola a vytvoření potřebných adresářů
    for directory in [TextConfig.INPUT_DIR, TextConfig.MODELS_DIR, TextConfig.DICTIONARY_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Vytvořen adresář: {directory}")

    # Kontrola vstupních dat
    if not any(f.endswith('.txt') for f in os.listdir(TextConfig.INPUT_DIR)):
        print(f"\nV adresáři {TextConfig.INPUT_DIR} nejsou žádné textové soubory.")
        print("Vložte textové soubory (.txt) a spusťte program znovu.")
        return

    # Inicializace
    processor = TextProcessor()
    model_handler = ModelHandler()

    # Pokud již existuje slovník, vypíšeme jeho statistiky
    if os.path.exists(os.path.join(TextConfig.DICTIONARY_DIR, TextConfig.VOCAB_FILENAME)):
        processor.print_dictionary_stats()
        rebuild_vocab = input("\nChcete přebudovat slovník? (ano/ne): ").lower() == 'ano'
    else:
        rebuild_vocab = True

    if rebuild_vocab:
        processor.build_vocabulary(TextConfig.INPUT_DIR)
        processor.print_dictionary_stats()

    # Načtení existujícího modelu nebo vytvoření nového
    try:
        weights, vocabulary = model_handler.load_best_model()
        print("\nNalezen existující model:")
        print(f"- Počet slov ve slovníku: {len(vocabulary)}")
        print(f"- Poslední loss: {weights['loss']:.4f}")
        print(f"- Epocha: {weights['epoch']}")

        # Kontrola kompatibility slovníků
        if len(vocabulary) != processor.unique_words:
            print("\nVAROVÁNÍ: Slovník modelu se neshoduje s aktuálním slovníkem!")
            print("Je potřeba přetrénovat model s novým slovníkem.")
            raise ValueError("Nekompatibilní slovník")

        processor.vocabulary = vocabulary
        model = MultiLayerWordEmbedding(
            vocab_size=len(vocabulary),
            embedding_dim=ModelConfig.EMBEDDING_DIM,
            hidden_layers=ModelConfig.HIDDEN_LAYERS,
            model_weights=weights
        )

        continue_training = input("\nChcete pokračovat v trénování? (ano/ne): ").lower()
        if continue_training != 'ano':
            interactive_mode(processor, model)
            return

    except (FileNotFoundError, ValueError) as e:
        print(f"\nVytváření nového modelu ({str(e)})")

        if processor.unique_words < 3:
            print("Nedostatek dat pro trénování.")
            return

        model = MultiLayerWordEmbedding(
            vocab_size=len(processor.vocabulary),
            embedding_dim=ModelConfig.EMBEDDING_DIM,
            hidden_layers=ModelConfig.HIDDEN_LAYERS
        )

    # Načtení trénovacích dat
    training_data = load_training_data(processor)

    if len(training_data) < ModelConfig.BATCH_SIZE:
        print("Nedostatek dat pro trénování.")
        return

    # Trénování
    train_model_incrementally(processor, model, training_data, model_handler)

    # Interaktivní režim
    interactive_mode(processor, model)


if __name__ == "__main__":
    main()