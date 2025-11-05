"""
Explanation Augmented (EA) Data Generator

Generates explanations for entity matching questions by using a local LLM hosted via vLLM.

Usage:
    python generate.py --config config.yaml
    
    # Or with verbose logging and dry run
    python generate.py --config config.yaml --verbose --dry-run
"""

import asyncio
import argparse
import logging
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import aiohttp
import pandas as pd
from tqdm import tqdm
import csv

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Configuration loaded from YAML file."""
    config_dict: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Filter out None values from overrides
        overrides = {k: v for k, v in overrides.items() if v is not None}
        config_dict.update(overrides)
        
        instance = cls(config_dict=config_dict)
        return instance
    
    def __getattr__(self, name: str) -> Any:
        """Allow dot notation access to config values."""
        if name == 'config_dict':
            return object.__getattribute__(self, name)
        return self.config_dict.get(name)
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        return f"Config({self.config_dict})"

# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(verbose: bool = False,
                  input_name: str = "") -> logging.Logger:
    """
    Configure logging for the classifier.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        input_name: Name of the input file for logging context
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # File handler - always logs DEBUG and above if verbose
    file_handler = logging.FileHandler(f'generate_{input_name}.log')
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler - only INFO and above (never DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ============================================================================
# Prompt Engineering
# ============================================================================

def load_prompt_template(prompt_path: str) -> str:
    """
    Load the few-shot prompt template from file.
    
    Args:
        prompt_path: Path to prompt template file
        
    Returns:
        Prompt template string
    """
    with open(prompt_path, 'r') as f:
        template = f.read().strip()
    logging.getLogger(__name__).debug(f"Loaded prompt template with {len(template)} characters")
    return template


def format_entity_row(row: pd.Series, suffix: str, base_columns: List[str], column_mapping: Dict[str, str]) -> str:
    """
    Format an entity row as a string with column names and values.
    
    Args:
        row: DataFrame row
        suffix: Column suffix ('_a' or '_b')
        base_columns: Base column names (without suffix)
        column_mapping: Dictionary mapping column names to prompt tags
        
    Returns:
        Formatted entity string like "[NAME] value [MANUFACTURER] value ..."
    """
    parts = []
    for col in base_columns:
        col_with_suffix = f"{col}{suffix}"
        if col_with_suffix in row.index:
            # Use mapped tag name if available, otherwise use column name
            tag_name = column_mapping.get(col.lower(), col).upper().replace('_', ' ')
            parts.append(f"[{tag_name}] {row[col_with_suffix]}")
    return " ".join(parts)


def create_prompt(prompt_template: str, row: pd.Series, base_columns: List[str], label: str, column_mapping: Dict[str, str]) -> str:
    """
    Create a complete prompt by appending the current example to the template.
    
    Args:
        prompt_template: Few-shot examples template
        row: DataFrame row with both _a and _b columns
        base_columns: Base column names (without suffix)
        label: "MATCH" or "NOT A MATCH"
        column_mapping: Dictionary mapping column names to prompt tags
        
    Returns:
        Complete prompt for generation
    """
    entity_a = format_entity_row(row, '_a', base_columns, column_mapping)
    entity_b = format_entity_row(row, '_b', base_columns, column_mapping)
    
    prompt = f"{prompt_template}\nEntity A: {entity_a}\nEntity B: {entity_b}\nLabel: {label}\nExplanation: "
    logging.getLogger(__name__).debug(f"Created prompt with {len(prompt)} characters for label={label}")
    return prompt

# ============================================================================
# Async Request Handler
# ============================================================================

async def generate_explanation(
    session: aiohttp.ClientSession,
    prompt: str,
    config: Config,
    logger: logging.Logger,
    retry_count: int = 0
) -> str:
    """
    Generate explanation for a single entity pair using vLLM.
    
    Args:
        session: aiohttp session
        prompt: Complete prompt string
        config: Configuration object
        logger: Logger instance
        retry_count: Current retry attempt
        
    Returns:
        Generated explanation or empty string on failure
    """
    url = f"{config.endpoint}/completions"
    
    payload = {
        "model": config.model,
        "prompt": prompt,
        "min_tokens": config.min_new_tokens,
        "max_tokens": config.max_new_tokens,
        "top_k": config.topk,
        "top_p": config.topp,
    }
    
    logger.debug(f"Sending request to {url} (retry={retry_count})")
    
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=config.timeout)
        ) as response:
            if response.status == 200:
                result = await response.json()
                explanation = result['choices'][0]['text'].strip()
                explanation = ' '.join(explanation.split()) # normalize whitespace to keep explanations single-line
                
                # Retry if explanation is empty (up to 3 attempts total)
                if not explanation and retry_count < 2:
                    logger.warning(f"Empty explanation received, retrying (attempt {retry_count + 2}/3)")
                    await asyncio.sleep(1)
                    return await generate_explanation(session, prompt, config, logger, retry_count + 1)
                
                return explanation
            else:
                error_text = await response.text()
                logger.warning(f"Request failed with status {response.status}: {error_text}")
                if retry_count < config.max_retries:
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    return await generate_explanation(session, prompt, config, logger, retry_count + 1)
                return ""
                
    except asyncio.TimeoutError:
        logger.warning(f"Request timeout (attempt {retry_count + 1}/{config.max_retries})")
        if retry_count < config.max_retries:
            await asyncio.sleep(2 ** retry_count)
            return await generate_explanation(session, prompt, config, logger, retry_count + 1)
        return ""
        
    except Exception as e:
        logger.error(f"Request failed: {e}")
        if retry_count < config.max_retries:
            await asyncio.sleep(2 ** retry_count)
            return await generate_explanation(session, prompt, config, logger, retry_count + 1)
        return ""

# ============================================================================
# Batch Processing
# ============================================================================

async def process_batch(
    session: aiohttp.ClientSession,
    batch_data: List[Tuple[int, str]],
    config: Config,
    logger: logging.Logger,
    semaphore: asyncio.Semaphore
) -> List[Tuple[int, str]]:
    """
    Process a batch of entity pairs concurrently.
    
    Args:
        session: aiohttp session
        batch_data: List of (index, prompt) tuples
        config: Configuration object
        logger: Logger instance
        semaphore: Semaphore for concurrency control
        
    Returns:
        List of (index, explanation) tuples
    """
    logger.debug(f"Processing batch of {len(batch_data)} items")
    
    async def process_single(idx: int, prompt: str) -> Tuple[int, str]:
        async with semaphore:
            explanation = await generate_explanation(session, prompt, config, logger)
            return (idx, explanation)
    
    tasks = [process_single(idx, prompt) for idx, prompt in batch_data]
    results = await asyncio.gather(*tasks)
    logger.debug(f"Completed batch of {len(batch_data)} items")
    return results


async def process_all_batches(
    df: pd.DataFrame,
    prompt_template: str,
    base_columns: List[str],
    config: Config,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Process all entity pairs in batches.
    
    Args:
        df: DataFrame with merged entity data
        prompt_template: Few-shot prompt template
        base_columns: Base column names (without suffix)
        config: Configuration object
        logger: Logger instance
        
    Returns:
        DataFrame with explanations added
    """
    # Get column mapping from config, default to empty dict
    column_mapping = config.column_mapping if config.column_mapping else {}
    
    # Prepare all prompts
    prompts = []
    for idx, row in df.iterrows():
        label = "MATCH" if row['label'] == 1 else "NOT A MATCH"
        prompt = create_prompt(prompt_template, row, base_columns, label, column_mapping)
        prompts.append((idx, prompt))
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(config.max_concurrent)
    
    # Process in batches
    results = {}
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(prompts), config.batch_size), desc="Processing batches"):
            batch = prompts[i:i + config.batch_size]
            batch_results = await process_batch(session, batch, config, logger, semaphore)
            
            for idx, explanation in batch_results:
                results[idx] = explanation
    
    # Add explanations to dataframe
    df['explanation'] = df.index.map(results)
    return df

# ============================================================================
# Data Loading and Processing
# ============================================================================

def load_and_merge_data(
    input_a_path: str,
    input_b_path: str,
    input_matches_path: str,
    logger: logging.Logger,
    dry_run: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and merge the three input CSV files.
    
    Args:
        input_a_path: Path to table A CSV
        input_b_path: Path to table B CSV
        input_matches_path: Path to matches CSV
        logger: Logger instance
        dry_run: If True, only process first 10 rows
        
    Returns:
        Tuple of (merged DataFrame, base column names)
    """
    logger.info("Loading input files...")
    
    # Load tables
    df_a = pd.read_csv(input_a_path)
    df_b = pd.read_csv(input_b_path)
    df_matches = pd.read_csv(input_matches_path)
    
    if dry_run:
        logger.info("DRY RUN: Processing first 10 rows only")
        df_matches = df_matches.head(10)
    
    logger.info(f"Loaded {len(df_a)} records from table A")
    logger.info(f"Loaded {len(df_b)} records from table B")
    logger.info(f"Loaded {len(df_matches)} match pairs")
    
    # Get base column names (excluding 'id')
    base_columns = [col for col in df_a.columns if col != 'id']
    logger.debug(f"Base columns: {base_columns}")
    
    # Merge data
    df_merged = df_matches.copy()
    
    # Merge with table A (adds columns with _a suffix)
    df_merged = df_merged.merge(
        df_a,
        left_on='ltable_id',
        right_on='id',
        suffixes=('', '_a')
    ).drop(columns=['id'])
    
    # Merge with table B (adds columns with _b suffix)
    df_merged = df_merged.merge(
        df_b,
        left_on='rtable_id',
        right_on='id',
        suffixes=('_a', '_b')
    ).drop(columns=['id'])
    
    logger.info(f"Merged {len(df_merged)} records for processing")
    logger.debug(f"Merged dataframe shape: {df_merged.shape}")
    
    return df_merged, base_columns

# ============================================================================
# Main Processing Pipeline
# ============================================================================

async def run_generation(
    input_a_path: str,
    input_b_path: str,
    input_matches_path: str,
    prompt_path: str,
    output_path: str,
    config: Config,
    logger: logging.Logger,
    dry_run: bool = False
) -> None:
    """
    Main processing pipeline.
    
    Args:
        input_a_path: Path to table A CSV
        input_b_path: Path to table B CSV
        input_matches_path: Path to matches CSV
        prompt_path: Path to prompt template
        output_path: Path to output CSV
        config: Configuration object
        logger: Logger instance
        dry_run: If True, only process first 10 rows
    """
    start_time = time.time()
    
    # Load and merge data
    df, base_columns = load_and_merge_data(
        input_a_path,
        input_b_path,
        input_matches_path,
        logger,
        dry_run
    )
    
    # Load prompt template
    logger.info(f"Loading prompt template from {prompt_path}")
    prompt_template = load_prompt_template(prompt_path)
    
    # Process all records
    logger.info("Starting explanation generation...")
    logger.debug(f"Concurrency: {config.max_concurrent}, Batch size: {config.batch_size}")
    df_with_explanations = await process_all_batches(
        df,
        prompt_template,
        base_columns,
        config,
        logger
    )
    
    # Select output columns
    output_columns = ['ltable_id', 'rtable_id', 'label', 'explanation']
    df_output = df_with_explanations[output_columns]
    
    # Check for empty explanations
    empty_count = (df_output['explanation'] == '').sum()
    if empty_count > 0:
        logger.warning(f"Found {empty_count} empty explanations in output")
    
    # Save results to CSV
    logger.info(f"Saving results to {output_path}")
    df_output.head(0).to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL) # write header first
    df_output.to_csv(output_path, index=False, mode='a', header=False, quoting=csv.QUOTE_NONNUMERIC) # append data with quoting handled
    logger.info(f"Successfully saved {len(df_output)} rows to {output_path}")
    
    elapsed = time.time() - start_time
    logger.debug(f"Total processing time: {elapsed:.2f} seconds ({elapsed/len(df_output):.2f}s per row)")

# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate explanation-augmented entity matching data using local LLM",
    )
    
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (overrides config file)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only first 10 rows for testing (overrides config file)"
    )
    
    parser.add_argument(
        "--verify-prompt",
        action="store_true",
        help="Display the complete prompt for the first element and exit"
    )
    
    parser.add_argument(
        "--verify-endpoint",
        action="store_true",
        help="Send prompt to vLLM endpoint and display response (requires --verify-prompt)"
    )
    
    return parser.parse_args()


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Validate config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        return
    
    # Load configuration from YAML
    try:
        config = Config.from_yaml(
            args.config,
            verbose=args.verbose if args.verbose else None,
            dry_run=args.dry_run if args.dry_run else None,
            verify_prompt=args.verify_prompt if args.verify_prompt else None,
            verify_endpoint=args.verify_endpoint if args.verify_endpoint else None
        )
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Setup logging
    input_name = Path(config.input_A).parent.name.lower() if config.input_A else "default"
    logger = setup_logging(verbose=config.verbose, 
                           input_name=input_name)
    
    logger.info(f"Starting EA Data Generation ({input_name})")
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Configuration: {config}")
    
    # Validate input files exist
    if not Path(config.input_A).exists():
        logger.error(f"Input A file not found: {config.input_A}")
        return
    elif not Path(config.input_B).exists():
        logger.error(f"Input B file not found: {config.input_B}")
        return
    elif not Path(config.input_matches).exists():
        logger.error(f"Input matches file not found: {config.input_matches}")
        return
    
    if config.prompt and not Path(config.prompt).exists():
        logger.error(f"Prompt file not found: {config.prompt}")
        return
    
    # Validate verify_endpoint requires verify_prompt
    if config.verify_endpoint and not config.verify_prompt:
        logger.error("--verify-endpoint requires --verify-prompt to be set")
        return
    
    # Handle verify_prompt mode
    if config.verify_prompt:
        logger.info("VERIFY PROMPT MODE: Displaying first prompt and exiting")
        
        # Load data (just first row)
        df_matches = pd.read_csv(config.input_matches).head(1)
        df_a = pd.read_csv(config.input_A)
        df_b = pd.read_csv(config.input_B)
        
        base_columns = [col for col in df_a.columns if col != 'id']
        
        # Merge first row
        df_merged = df_matches.merge(
            df_a,
            left_on='ltable_id',
            right_on='id',
            suffixes=('', '_a')
        ).drop(columns=['id'])
        
        df_merged = df_merged.merge(
            df_b,
            left_on='rtable_id',
            right_on='id',
            suffixes=('_a', '_b')
        ).drop(columns=['id'])
        
        # Load prompt template
        prompt_template = load_prompt_template(config.prompt)
        
        # Get column mapping from config
        column_mapping = config.column_mapping if config.column_mapping else {}
        
        # Create prompt for first row
        first_row = df_merged.iloc[0]
        label = "MATCH" if first_row['label'] == 1 else "NOT A MATCH"
        complete_prompt = create_prompt(prompt_template, first_row, base_columns, label, column_mapping)
        
        # Display with clear formatting
        print("\n" + "="*80)
        print("COMPLETE PROMPT FOR FIRST ELEMENT")
        print("="*80)
        print(complete_prompt)
        print("="*80 + "\n")
        
        # Handle verify_endpoint mode
        if config.verify_endpoint:
            logger.info("VERIFY ENDPOINT MODE: Sending prompt to vLLM and displaying response")
            
            async with aiohttp.ClientSession() as session:
                try:
                    explanation = await generate_explanation(
                        session,
                        complete_prompt,
                        config,
                        logger
                    )
                    
                    print("\n" + "="*80)
                    print("VLLM RESPONSE")
                    print("="*80)
                    if explanation:
                        print(explanation)
                    else:
                        print("(No response or empty response)")
                    print("="*80 + "\n")
                    
                    logger.info("Endpoint verification complete")
                    
                except Exception as e:
                    logger.error(f"Endpoint verification failed: {e}")
                    print(f"\nERROR: Failed to connect to endpoint: {e}\n")
        
        return
    
    # Run generation
    try:
        await run_generation(
            input_a_path=config.input_A,
            input_b_path=config.input_B,
            input_matches_path=config.input_matches,
            prompt_path=config.prompt,
            output_path=config.output,
            config=config,
            logger=logger,
            dry_run=config.dry_run
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
    
    logger.info("Generation complete!")

if __name__ == "__main__":
    asyncio.run(main())