"""
Inference pipeline for running predictions with pretrained models.
"""

import torch
import logging
from typing import List, Dict, Optional
from src.models.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Pipeline for running inference with transformer models."""

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        max_length: int = 100,
    ):
        """
        Initialize the inference pipeline.

        Args:
            model_name: Hugging Face model identifier
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum length of generated text
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        # Load model and tokenizer
        self.loader = ModelLoader(model_name, device=self.device)
        self.model, self.tokenizer = self.loader.load_model_and_tokenizer()
        
        logger.info(f"InferencePipeline initialized with model: {model_name}")

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text based on a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated text sequences
        """
        try:
            logger.info(f"Generating text for prompt: {prompt[:50]}...")
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(  # type: ignore
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                )
            
            # Decode outputs
            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            logger.info(f"Generated {len(generated_texts)} sequence(s)")
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise

    def classify_text(self, text: str) -> Dict[str, float]:
        """
        Classify text using the model (for classification models).

        Args:
            text: Input text to classify

        Returns:
            Dictionary with predictions and confidence scores
        """
        try:
            logger.info(f"Classifying text: {text[:50]}...")
            
            # Tokenize input
            inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            result = {
                "text": text,
                "predictions": probabilities[0].cpu().numpy().tolist(),
            }
            
            logger.info("Classification completed")
            return result
            
        except Exception as e:
            logger.error(f"Error during classification: {str(e)}")
            raise

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments for generate_text

        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            generated = self.generate_text(
                prompt,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            results.extend(generated)
        
        return results

    def get_pipeline_info(self) -> Dict:
        """Get information about the inference pipeline."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "model_info": self.loader.get_model_info(),
        }


def run_inference(
    prompt: str,
    model_name: str = "gpt2",
    max_new_tokens: int = 50,
) -> str:
    """
    Convenience function to run inference on a prompt.

    Args:
        prompt: Input text prompt
        model_name: Hugging Face model identifier
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated text
    """
    pipeline = InferencePipeline(model_name)
    results = pipeline.generate_text(prompt, max_new_tokens=max_new_tokens)
    return results[0] if results else ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    pipeline = InferencePipeline("gpt2")
    
    # Generate text
    prompt = "The future of artificial intelligence is"
    results = pipeline.generate_text(prompt, max_new_tokens=50)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {results[0]}")
    
    # Print pipeline info
    print(f"\nPipeline Info: {pipeline.get_pipeline_info()}")
