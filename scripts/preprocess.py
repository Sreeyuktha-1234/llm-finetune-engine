"""
Preprocessing script for preparing datasets for model fine-tuning.
Handles data loading, cleaning, tokenization, and formatting.
"""

import json
import logging
import os
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess raw datasets for LLM fine-tuning."""

    def __init__(
        self,
        raw_data_path: str = "data/raw",
        processed_data_path: str = "data/processed",
    ):
        """
        Initialize the preprocessor.

        Args:
            raw_data_path: Path to raw data directory
            processed_data_path: Path to save processed data
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
        # Create processed data directory if it doesn't exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataPreprocessor initialized")
        logger.info(f"Raw data path: {self.raw_data_path}")
        logger.info(f"Processed data path: {self.processed_data_path}")

    def load_raw_data(self, filename: str = "dataset.json") -> Dict:
        """
        Load raw dataset from JSON file.

        Args:
            filename: Name of the raw data file

        Returns:
            Dictionary containing the raw data
        """
        file_path = self.raw_data_path / filename
        
        try:
            logger.info(f"Loading raw data from: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {len(data.get('data', []))} samples")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file: {file_path}")
            raise

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and normalizing.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def preprocess_samples(self, data: Dict) -> List[Dict]:
        """
        Preprocess individual samples from the dataset.

        Args:
            data: Raw dataset dictionary

        Returns:
            List of preprocessed samples
        """
        processed_samples = []
        
        for sample in data.get('data', []):
            processed_sample = {
                'id': sample.get('id'),
                'text': self.clean_text(sample.get('text', '')),
                'category': sample.get('category', 'unknown'),
                'original_length': sample.get('length', 0),
                'processed_length': len(self.clean_text(sample.get('text', '')).split()),
            }
            processed_samples.append(processed_sample)
        
        logger.info(f"Preprocessed {len(processed_samples)} samples")
        return processed_samples

    def split_by_category(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group samples by category.

        Args:
            samples: List of preprocessed samples

        Returns:
            Dictionary mapping categories to sample lists
        """
        categorized = {}
        
        for sample in samples:
            category = sample.get('category', 'unknown')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(sample)
        
        logger.info(f"Grouped samples into {len(categorized)} categories")
        for category, items in categorized.items():
            logger.info(f"  {category}: {len(items)} samples")
        
        return categorized

    def save_processed_data(
        self,
        data: Dict,
        output_filename: str = "processed_dataset.json"
    ) -> None:
        """
        Save processed data to JSON file.

        Args:
            data: Processed data to save
            output_filename: Name of the output file
        """
        output_path = self.processed_data_path / output_filename
        
        try:
            logger.info(f"Saving processed data to: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved processed data")
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
            raise

    def preprocess_pipeline(self) -> Dict:
        """
        Run the complete preprocessing pipeline.

        Returns:
            Dictionary containing all processed data
        """
        try:
            # Load raw data
            raw_data = self.load_raw_data()
            
            # Preprocess samples
            processed_samples = self.preprocess_samples(raw_data)
            
            # Split by category
            categorized = self.split_by_category(processed_samples)
            
            # Prepare output
            output_data = {
                'metadata': raw_data.get('metadata', {}),
                'processed_samples': processed_samples,
                'by_category': categorized,
                'splits': raw_data.get('splits', {}),
                'preprocessing_stats': {
                    'total_samples': len(processed_samples),
                    'categories': len(categorized),
                    'avg_tokens_per_sample': sum(s['processed_length'] for s in processed_samples) / len(processed_samples) if processed_samples else 0,
                }
            }
            
            # Save processed data
            self.save_processed_data(output_data)
            
            logger.info("Preprocessing pipeline completed successfully")
            return output_data
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point for preprocessing script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        raw_data_path="data/raw",
        processed_data_path="data/processed"
    )
    
    # Run preprocessing pipeline
    result = preprocessor.preprocess_pipeline()
    
    # Print summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Total samples processed: {result['preprocessing_stats']['total_samples']}")
    print(f"Number of categories: {result['preprocessing_stats']['categories']}")
    print(f"Average tokens per sample: {result['preprocessing_stats']['avg_tokens_per_sample']:.2f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
