"""
Vícevrstvá neuronová síť pro word embeddings s detailním vysvětlením backpropagace.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import time

class Layer:
    """
    Základní třída pro vrstvu neuronové sítě.
    Každá vrstva si pamatuje své vstupy, výstupy a gradienty pro backpropagaci.
    """
    def __init__(self, input_size: int, output_size: int):
        # Xavier/Glorot inicializace vah
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros(output_size)

        # Úložiště pro backpropagaci
        self.input = None
        self.output = None
        self.input_gradient = None
        self.weight_gradient = None
        self.bias_gradient = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Dopředný průchod vrstvou.
        Uloží vstup pro pozdější použití v backpropagaci.
        """
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Zpětný průchod vrstvou (backpropagace).

        Args:
            output_gradient: Gradient z následující vrstvy

        Returns:
            Gradient pro předchozí vrstvu
        """
        # Výpočet gradientů
        self.weight_gradient = np.outer(self.input, output_gradient)
        self.bias_gradient = output_gradient
        self.input_gradient = np.dot(output_gradient, self.weights.T)
        return self.input_gradient

    def update(self, learning_rate: float):
        """
        Aktualizace vah a biasů pomocí vypočtených gradientů.
        """
        self.weights -= learning_rate * self.weight_gradient
        self.bias -= learning_rate * self.bias_gradient

class ActivationLayer:
    """
    Aktivační vrstva aplikující nelineární funkci.
    Podporuje ReLU a tanh aktivace.
    """
    def __init__(self, activation_type: str = 'relu'):
        self.activation_type = activation_type
        self.input = None
        self.output = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        if self.activation_type == 'relu':
            self.output = np.maximum(0, input_data)
        elif self.activation_type == 'tanh':
            self.output = np.tanh(input_data)
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        if self.activation_type == 'relu':
            return output_gradient * (self.input > 0)
        elif self.activation_type == 'tanh':
            return output_gradient * (1 - np.tanh(self.input) ** 2)

class MultiLayerWordEmbedding:
    """
    Vícevrstvá neuronová síť pro word embeddings.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 100,
                 hidden_layers: List[int] = [64, 32],
                 activation: str = 'relu',
                 model_weights: Optional[Dict[str, Any]] = None):
        """
        Inicializace modelu s možností načtení existujících vah.

        Args:
            vocab_size: Velikost slovníku
            embedding_dim: Dimenze embedding vektorů
            hidden_layers: Seznam velikostí skrytých vrstev
            activation: Typ aktivační funkce ('relu' nebo 'tanh')
            model_weights: Slovník s předtrénovanými vahami, nebo None pro nový model
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Inicializace vrstev
        self.embedding_layer = Layer(vocab_size, embedding_dim)

        self.layers = []
        current_dim = embedding_dim
        for hidden_dim in hidden_layers:
            self.layers.append(Layer(current_dim, hidden_dim))
            self.layers.append(ActivationLayer(activation))
            current_dim = hidden_dim

        self.output_layer = Layer(current_dim, vocab_size)

        # Načtení vah, pokud jsou k dispozici
        if model_weights is not None:
            self.load_weights(model_weights)

        # Statistiky trénování
        self.training_stats = defaultdict(list)
        self.epoch_stats = defaultdict(float)
        self.best_loss = float('inf')

    def load_weights(self, weights: Dict[str, Any]):
        """Načte váhy do modelu."""
        self.embedding_layer.weights = weights['embedding_weights']

        hidden_weights = weights['hidden_weights']
        hidden_biases = weights['hidden_biases']

        weight_idx = 0
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.weights = hidden_weights[weight_idx]
                layer.bias = hidden_biases[weight_idx]
                weight_idx += 1

        self.output_layer.weights = weights['output_weights']
        self.output_layer.bias = weights['output_bias']

    def forward(self, word_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dopředný průchod sítí.

        Args:
            word_indices: Seznam indexů kontextových slov

        Returns:
            Tuple[np.ndarray, np.ndarray]: (embedding vektor, výstupní pravděpodobnosti)
        """
        # Průměrování embedding vektorů kontextových slov
        context_vectors = [self.embedding_layer.weights[idx] for idx in word_indices]
        embedding = np.mean(context_vectors, axis=0)

        # Průchod skrytými vrstvami
        activation = embedding
        for layer in self.layers:
            activation = layer.forward(activation)

        # Výstupní vrstva a softmax
        logits = self.output_layer.forward(activation)
        probs = self.softmax(logits)

        return embedding, probs

    def backward(self, word_indices: List[int], target_idx: int, learning_rate: float):
        """
        Zpětný průchod sítí (backpropagace).

        Args:
            word_indices: Seznam indexů kontextových slov
            target_idx: Index cílového slova
            learning_rate: Rychlost učení
        """
        # Forward pass
        embedding, probs = self.forward(word_indices)

        # Výpočet počáteční chyby (cross-entropy gradient)
        target_one_hot = np.zeros(self.vocab_size)
        target_one_hot[target_idx] = 1
        output_gradient = probs - target_one_hot

        # Backpropagace přes výstupní vrstvu
        gradient = self.output_layer.backward(output_gradient)

        # Backpropagace přes skryté vrstvy
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        # Aktualizace vah
        self.output_layer.update(learning_rate)
        for layer in self.layers:
            if isinstance(layer, Layer):  # Přeskočení aktivačních vrstev
                layer.update(learning_rate)

        # Aktualizace embedding vektorů
        embedding_gradient = gradient
        for idx in word_indices:
            self.embedding_layer.weights[idx] -= learning_rate * embedding_gradient / len(word_indices)

        # Uložení statistik
        loss = -np.sum(target_one_hot * np.log(probs + 1e-10))
        self.epoch_stats['loss'] += loss
        self.epoch_stats['samples'] += 1

    def end_epoch(self):
        """Zpracování statistik na konci epochy."""
        avg_loss = self.epoch_stats['loss'] / max(1, self.epoch_stats['samples'])
        self.training_stats['loss'].append(avg_loss)
        print(f"Průměrná ztráta: {avg_loss:.4f}")

        # Reset statistik
        self.epoch_stats = defaultdict(float)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Výpočet softmax funkce s numerickou stabilitou."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def get_similar_words(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Nalezení nejpodobnějších slov podle embedding vektoru.

        Args:
            embedding: Vstupní embedding vektor
            top_k: Počet nejpodobnějších slov k nalezení

        Returns:
            List[Tuple[int, float]]: Seznam (index slova, podobnost)
        """
        # Kosinová podobnost mezi vstupním vektorem a všemi embeddingy
        similarities = np.dot(self.embedding_layer.weights, embedding) / (
            np.linalg.norm(self.embedding_layer.weights, axis=1) * np.linalg.norm(embedding)
        )
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]

    def print_training_progress(self, epoch: int, batch: int, total_batches: int):
        """
        Výpis průběhu trénování.

        Args:
            epoch: Číslo současné epochy
            batch: Číslo současného batche
            total_batches: Celkový počet batchů
        """
        avg_loss = self.epoch_stats['loss'] / max(1, self.epoch_stats['samples'])
        progress = batch / total_batches * 100
        print(f"\rEpocha {epoch+1}: {progress:.1f}% [Ztráta: {avg_loss:.4f}]", end="")
