"""
Modul pro správu ukládání a načítání modelů.
"""

import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from config import TextConfig, ModelManager

class ModelHandler:
    def __init__(self, model_dir: str = TextConfig.MODELS_DIR):
        """Inicializace handleru pro správu modelů."""
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def save_model(self,
                  model: 'MultiLayerWordEmbedding',
                  loss: float,
                  epoch: int,
                  vocab: list) -> str:
        """
        Uloží model včetně všech potřebných informací.

        Args:
            model: Instance modelu k uložení
            loss: Aktuální hodnota loss
            epoch: Číslo epochy
            vocab: Seznam slov ve slovníku

        Returns:
            str: Název souboru, pod kterým byl model uložen
        """
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = ModelManager.get_model_filename(loss, epoch, timestamp)
        filepath = os.path.join(self.model_dir, filename)

        # Získání vah skrytých vrstev a jejich biasů
        hidden_weights = []
        hidden_biases = []
        for layer in model.layers:
            if hasattr(layer, 'weights'):
                hidden_weights.append(layer.weights)
                hidden_biases.append(layer.bias)

        # Převod seznamů na numpy objekty
        hidden_weights_array = np.array(hidden_weights, dtype=object)
        hidden_biases_array = np.array(hidden_biases, dtype=object)

        # Uložení vah a dalších informací
        np.savez(
            filepath,
            embedding_weights=model.embedding_layer.weights,
            hidden_weights=hidden_weights_array,
            hidden_biases=hidden_biases_array,
            output_weights=model.output_layer.weights,
            output_bias=model.output_layer.bias,
            vocabulary=np.array(vocab, dtype=object),
            loss=np.array([loss]),
            epoch=np.array([epoch])
        )

        return filename

    def load_model(self,
                  filename: Optional[str] = None) -> Tuple[Dict[str, Any], list]:
        """
        Načte model ze souboru.

        Args:
            filename: Název souboru k načtení, nebo None pro načtení nejlepšího modelu

        Returns:
            Tuple[Dict[str, Any], list]: (Slovník s vahami a parametry, seznam slov)
        """
        if filename is None:
            filename = ModelManager.get_best_model()
            if filename is None:
                raise ValueError("Nebyl nalezen žádný uložený model.")

        filepath = os.path.join(self.model_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model {filename} nebyl nalezen.")

        data = np.load(filepath, allow_pickle=True)

        return {
            'embedding_weights': data['embedding_weights'],
            'hidden_weights': data['hidden_weights'].tolist(),  # Převod zpět na seznam
            'hidden_biases': data['hidden_biases'].tolist(),   # Převod zpět na seznam
            'output_weights': data['output_weights'],
            'output_bias': data['output_bias'],
            'loss': float(data['loss'][0]),
            'epoch': int(data['epoch'][0])
        }, data['vocabulary'].tolist()

    def load_best_model(self) -> Tuple[Dict[str, Any], list]:
        """Načte nejlepší dostupný model podle hodnoty loss."""
        return self.load_model(None)