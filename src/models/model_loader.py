"""
Model loader module for loading pretrained models from Hugging Face.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage pretrained transformer models."""

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """
        Initialize the model loader.

        Args:
            model_name: Hugging Face model identifier (default: "gpt2")
            device: Device to load model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        logger.info(f"ModelLoader initialized. Device: {self.device}")

    def load_model(self):
        """
        Load pretrained model from Hugging Face.

        Returns:
            The loaded model
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            )
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise

    def load_tokenizer(self):
        """
        Load pretrained tokenizer from Hugging Face.

        Returns:
            The loaded tokenizer
        """
        try:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded successfully")
            return self.tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer {self.model_name}: {str(e)}")
            raise

    def load_model_and_tokenizer(self):
        """
        Load both model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        return model, tokenizer

    def get_model_info(self):
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "Model not loaded"}

        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_type": type(self.model).__name__,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }


def load_pretrained_model(model_name: str = "gpt2"):
    """
    Convenience function to load a pretrained model and tokenizer.

    Args:
        model_name: Hugging Face model identifier

    Returns:
        Tuple of (model, tokenizer)
    """
    loader = ModelLoader(model_name)
    return loader.load_model_and_tokenizer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    loader = ModelLoader("gpt2")
    model, tokenizer = loader.load_model_and_tokenizer()
    print(loader.get_model_info())
