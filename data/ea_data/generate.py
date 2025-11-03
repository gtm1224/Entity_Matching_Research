"""
Explanation Augmented (EA) Data Generator

Generates explanations for entity matching questions by using
a local LLM hosted via vLLM.

Usage:
    python generate.py --input <path> --prompt <path> --output <path>

    # Basic usage
    python generate.py \
        --input /home/danielruiz/workspace/phd/Entity_Matching_Research/data/Beer/train.csv \
        --prompt /home/danielruiz/workspace/phd/Entity_Matching_Research/data/ea_data/prompts/beer.txt \
        --output /home/danielruiz/workspace/phd/Entity_Matching_Research/data/ea_data/beer_train_ea.csv

    # Advanced usage with custom configuration
    python generate.py \
        --input /home/danielruiz/workspace/phd/Entity_Matching_Research/data/Beer/train.csv \
        --prompt /home/danielruiz/workspace/phd/Entity_Matching_Research/data/ea_data/prompts/beer.txt \
        --output /home/danielruiz/workspace/phd/Entity_Matching_Research/data/ea_data/beer_train_ea.csv \
        --endpoint http://localhost:8000/v1 \
        --batch-size 100 \
        --max-concurrent 20 \
        --verbose
"""

import asyncio
import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import aiohttp
import pandas as pd
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Configuration for spacing issue classifier."""
    vllm_endpoint: str = "http://localhost:8000/v1"
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    batch_size: int = 50
    max_concurrent: int = 10
    timeout: int = 30
    max_retries: int = 3
    # TODO: CONTINUE FROM HERE
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("Batch size must be >= 1")
        if self.max_concurrent < 1:
            raise ValueError("Max concurrent requests must be >= 1")
        if self.timeout < 1:
            raise ValueError("Timeout must be >= 1 second")


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure logging for the classifier.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('spacing_classifier.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# ============================================================================
# Prompt Engineering
# ============================================================================

def build_classification_prompt(question: str) -> List[Dict[str, str]]:
    """
    Build few-shot prompt for spacing issue classification.
    
    Args:
        question: The question text to classify
        
    Returns:
        List of message dicts for OpenAI chat completion format
    """
    system_prompt = """You are a text quality classifier that identifies spacing issues in text. Your task is to detect when words are concatenated without proper spaces."""
    
    user_prompt = f"""Classify if the following text has spacing issues (words concatenated without spaces).

Examples WITH spacing issues (respond with "1"):
- "enormousmulticellularseaweeds"
- "Drosophilamelanogastercultures"
- "betweenapoenzymesandcofactors"
- "Salmonellatyphimuriumhas"
- "Comparephototrophsandchemotrophs"
- "ofthe"
- "Whatisthebasic"

Examples WITHOUT spacing issues (respond with "0"):
- "What is the basic unit of life?"
- "Compare phototrophs and chemotrophs."
- "Salmonella typhimurium has flagella."
- "What are the characteristics of seaweeds?"
- "The relationship between apoenzymes and cofactors"

Text to classify: "{question}"

Respond with only: 0 or 1"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


# ============================================================================
# Async Request Handler
# ============================================================================

async def classify_question_async(
    session: aiohttp.ClientSession,
    question: str,
    index: int,
    config: Config,
    logger: logging.Logger,
    semaphore: asyncio.Semaphore
) -> Tuple[int, int]:
    """
    Classify a single question using the LLM API with retry logic.
    Uses vLLM's structured outputs for guaranteed binary response.
    
    Args:
        session: aiohttp client session
        question: Question text to classify
        index: Original index in dataset
        config: Configuration object
        logger: Logger instance
        semaphore: Semaphore for concurrency control
        
    Returns:
        Tuple of (index, classification_result)
    """
    # Handle edge cases
    if not question or pd.isna(question):
        return (index, 0)
    
    if not isinstance(question, str):
        question = str(question)
    
    if len(question.strip()) == 0:
        return (index, 0)
    
    async with semaphore:
        for attempt in range(config.max_retries):
            try:
                messages = build_classification_prompt(question)
                
                # Use vLLM structured outputs with choice constraint
                payload = {
                    "model": config.model_name,
                    "messages": messages,
                    "temperature": config.temperature,
                    "max_tokens": 5,
                    "extra_body": {
                        "guided_choice": ["0", "1"],
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                }
                
                timeout = aiohttp.ClientTimeout(total=config.timeout)
                
                async with session.post(
                    f"{config.vllm_endpoint}/chat/completions",
                    json=payload,
                    timeout=timeout
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Extract response text - guaranteed to be "0" or "1" as string
                    content = data['choices'][0]['message']['content'].strip()
                    
                    # Convert string to int for storage
                    if content == "0":
                        classification = 0
                    elif content == "1":
                        classification = 1
                    else:
                        logger.warning(
                            f"Unexpected value '{content}' for index {index}, treating as -1. "
                            f"Full response: {data}"
                        )
                        classification = -1
                    
                    return (index, classification)
                    
            except asyncio.TimeoutError:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Timeout for index {index}, attempt {attempt + 1}/{config.max_retries}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
                
            except aiohttp.ClientError as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"API error for index {index}, attempt {attempt + 1}/{config.max_retries}: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Unexpected error for index {index}: {e}", exc_info=True)
                return (index, -1)
        
        # All retries exhausted
        logger.error(f"Failed to classify index {index} after {config.max_retries} attempts")
        return (index, -1)


# ============================================================================
# Batch Processing
# ============================================================================

async def process_batch(
    questions: List[str],
    start_idx: int,
    config: Config,
    logger: logging.Logger
) -> Dict[int, int]:
    """
    Process a batch of questions concurrently.
    
    Args:
        questions: List of question texts
        start_idx: Starting index for this batch
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Dictionary mapping index to classification result
    """
    semaphore = asyncio.Semaphore(config.max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            classify_question_async(
                session, question, start_idx + i, config, logger, semaphore
            )
            for i, question in enumerate(questions)
        ]
        
        results = await asyncio.gather(*tasks)
        
    return dict(results)


# ============================================================================
# Main Processing Pipeline
# ============================================================================

async def classify_dataset(
    input_path: str,
    output_path: str,
    config: Config,
    logger: logging.Logger,
    dry_run: bool = False
) -> None:
    """
    Main pipeline for classifying spacing issues in dataset.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        config: Configuration object
        logger: Logger instance
        dry_run: If True, process only first 10 rows
    """
    logger.info(f"Loading dataset from {input_path}")
    
    # Load parquet file
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        raise
    
    # Validate required column
    if 'question' not in df.columns:
        raise ValueError("Input file must contain 'question' column")
    
    # Handle dry run
    if dry_run:
        logger.info("DRY RUN MODE: Processing only first 10 rows")
        df = df.head(10)
    
    total_questions = len(df)
    logger.info(f"Processing {total_questions} questions")
    
    if total_questions == 0:
        logger.warning("Empty dataset, creating output with all 0s")
        df['spacing_issues'] = 0
        df.to_parquet(output_path, index=False)
        return
    
    # Process in batches
    all_classifications = {}
    start_time = time.time()
    
    with tqdm(total=total_questions, desc="Classifying questions") as pbar:
        for batch_start in range(0, total_questions, config.batch_size):
            batch_end = min(batch_start + config.batch_size, total_questions)
            batch_questions = df.iloc[batch_start:batch_end]['question'].tolist()
            
            batch_results = await process_batch(
                batch_questions, batch_start, config, logger
            )
            
            all_classifications.update(batch_results)
            pbar.update(len(batch_questions))
    
    # Add classification column
    df['spacing_issues'] = df.index.map(all_classifications)
    
    # Save to output parquet
    try:
        df.to_parquet(output_path, index=False)
        logger.info(f"Successfully saved results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save output file: {e}")
        raise
    
    # Report statistics
    elapsed_time = time.time() - start_time
    spacing_issues_count = (df['spacing_issues'] == 1).sum()
    failed_count = (df['spacing_issues'] == -1).sum()
    
    logger.info("\n" + "="*60)
    logger.info("CLASSIFICATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total questions processed: {total_questions}")
    logger.info(f"Spacing issues detected: {spacing_issues_count} ({spacing_issues_count/total_questions*100:.2f}%)")
    logger.info(f"Failed classifications: {failed_count} ({failed_count/total_questions*100:.2f}%)")
    logger.info(f"Processing time: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per request: {elapsed_time/total_questions:.2f} seconds")
    logger.info("="*60)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_output(output_path: str, original_row_count: int, logger: logging.Logger) -> bool:
    """
    Validate the output parquet file.
    
    Args:
        output_path: Path to output file
        original_row_count: Expected number of rows
        logger: Logger instance
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        df = pd.read_parquet(output_path)
        
        # Check row count
        if len(df) != original_row_count:
            logger.error(f"Row count mismatch: expected {original_row_count}, got {len(df)}")
            return False
        
        # Check for spacing_issues column
        if 'spacing_issues' not in df.columns:
            logger.error("Missing 'spacing_issues' column")
            return False
        
        # Check valid values
        valid_values = df['spacing_issues'].isin([0, 1, -1]).all()
        if not valid_values:
            logger.error("Invalid values in 'spacing_issues' column")
            return False
        
        logger.info("Output validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def display_sample_results(df: pd.DataFrame, n: int = 10) -> None:
    """
    Display random samples for manual verification.
    
    Args:
        df: DataFrame with classifications
        n: Number of samples to display
    """
    print("\n" + "="*80)
    print("SAMPLE RESULTS FOR MANUAL VERIFICATION")
    print("="*80)
    
    sample = df.sample(min(n, len(df)))
    
    for idx, row in sample.iterrows():
        label = row.get('spacing_issues', -1)
        label_str = {0: "NO ISSUES", 1: "SPACING ISSUES", -1: "FAILED"}[label]
        
        print(f"\n[{label_str}] Question:")
        print(f"  {row['question'][:200]}...")
        print(f"  (Index: {idx})")
    
    print("\n" + "="*80)


# ============================================================================
# Testing Functions
# ============================================================================

def test_classifier():
    """Run unit tests on key functions."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    # Test prompt construction
    print("\nTesting prompt construction:")
    test_question = "Whatisthebasicunit"
    messages = build_classification_prompt(test_question)
    assert len(messages) == 2, "Should have system and user messages"
    assert messages[0]["role"] == "system", "First message should be system"
    assert messages[1]["role"] == "user", "Second message should be user"
    assert test_question in messages[1]["content"], "Question should be in user message"
    print("  ✓ Prompt construction working correctly")
    
    print("\nNote: Response parsing is now handled by vLLM structured outputs")
    print("  ✓ Using guided_choice=['0', '1'] for guaranteed binary response")
    
    print("\n" + "="*60)
    print("UNIT TESTS COMPLETE")
    print("="*60)


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify spacing issues in MMLU-ProX questions using local LLM"
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input parquet file"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output parquet file"
    )
    
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1",
        help="vLLM endpoint URL (default: http://localhost:8000/v1)"
    )
    
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only first 10 rows for testing"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run unit tests and exit"
    )
    
    return parser.parse_args()


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Run tests if requested
    if args.test:
        test_classifier()
        return
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Create configuration
    config = Config(
        vllm_endpoint=args.endpoint,
        model_name=args.model,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
    )
    
    logger.info("Starting MMLU-ProX Spacing Issue Classifier")
    logger.info(f"Configuration: {config}")
    
    # Validate input file exists
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Get original row count for validation
    original_df = pd.read_parquet(args.input)
    original_row_count = len(original_df)
    
    # Run classification
    try:
        await classify_dataset(
            args.input,
            args.output,
            config,
            logger,
            dry_run=args.dry_run
        )
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise
    
    # Validate output
    if not args.dry_run:
        validate_output(args.output, original_row_count, logger)
    
    # # Display sample results
    # output_df = pd.read_parquet(args.output)
    # display_sample_results(output_df, n=10)
    
    logger.info("Classification complete!")


if __name__ == "__main__":
    asyncio.run(main())