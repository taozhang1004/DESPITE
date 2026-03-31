#!/usr/bin/env python3
"""
Cost Calculation Tool - Calculate API costs for successful tasks

This tool:
- Scans multiple task directories for result.json files
- Extracts token usage information from each task
- Calculates total cost based on cache hit/miss and output token pricing
- Computes average cost per successful task
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TokenUsage:
    """Token usage statistics for a single task"""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


@dataclass
class CostBreakdown:
    """Cost breakdown for token usage"""
    cache_hit_cost: float = 0.0
    cache_miss_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    def __add__(self, other: 'CostBreakdown') -> 'CostBreakdown':
        """Add two cost breakdowns together"""
        return CostBreakdown(
            cache_hit_cost=self.cache_hit_cost + other.cache_hit_cost,
            cache_miss_cost=self.cache_miss_cost + other.cache_miss_cost,
            output_cost=self.output_cost + other.output_cost,
            total_cost=self.total_cost + other.total_cost
        )


class CostCalculator:
    """Calculates API costs from token usage data"""

    # Pricing per 1M tokens
    CACHE_HIT_PRICE_PER_1M = 0.028
    CACHE_MISS_PRICE_PER_1M = 0.28
    OUTPUT_PRICE_PER_1M = 0.42

    def __init__(self):
        self.total_tokens = TokenUsage()
        self.total_cost = CostBreakdown()
        self.processed_tasks = 0
        self.failed_tasks = 0

    def calculate_task_cost(self, token_usage: TokenUsage) -> CostBreakdown:
        """Calculate cost for a single task's token usage"""
        # Convert tokens to millions for pricing calculation
        cache_hit_millions = token_usage.cache_hit_tokens / 1_000_000
        cache_miss_millions = token_usage.cache_miss_tokens / 1_000_000
        output_millions = token_usage.total_completion_tokens / 1_000_000

        cache_hit_cost = cache_hit_millions * self.CACHE_HIT_PRICE_PER_1M
        cache_miss_cost = cache_miss_millions * self.CACHE_MISS_PRICE_PER_1M
        output_cost = output_millions * self.OUTPUT_PRICE_PER_1M

        total_cost = cache_hit_cost + cache_miss_cost + output_cost

        return CostBreakdown(
            cache_hit_cost=cache_hit_cost,
            cache_miss_cost=cache_miss_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )

    def process_result_file(self, result_file: Path) -> Optional[TokenUsage]:
        """Extract token usage from a result.json file"""
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Navigate to token_usage in codegen_metadata
            codegen_metadata = data.get('codegen_metadata', {})
            token_usage_data = codegen_metadata.get('token_usage', {})

            if not token_usage_data:
                return None

            return TokenUsage(
                total_prompt_tokens=token_usage_data.get('total_prompt_tokens', 0),
                total_completion_tokens=token_usage_data.get('total_completion_tokens', 0),
                total_tokens=token_usage_data.get('total_tokens', 0),
                cache_hit_tokens=token_usage_data.get('cache_hit_tokens', 0),
                cache_miss_tokens=token_usage_data.get('cache_miss_tokens', 0)
            )

        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error processing {result_file}: {e}")
            return None

    def scan_directory(self, directory: Path, verbose: bool = False) -> None:
        """Scan a directory for result.json files and accumulate costs"""
        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist")
            return

        # Find all result.json files recursively in nested subdirectories
        result_files = list(directory.glob("**/result.json"))
        
        if verbose:
            print(f"Scanning {directory}: Found {len(result_files)} result.json files")

        for result_file in result_files:
            token_usage = self.process_result_file(result_file)
            
            if token_usage is None:
                self.failed_tasks += 1
                if verbose:
                    print(f"  Skipped {result_file.parent.name}: No token usage data")
                continue

            # Calculate cost for this task
            task_cost = self.calculate_task_cost(token_usage)

            # Accumulate totals
            self.total_tokens.total_prompt_tokens += token_usage.total_prompt_tokens
            self.total_tokens.total_completion_tokens += token_usage.total_completion_tokens
            self.total_tokens.total_tokens += token_usage.total_tokens
            self.total_tokens.cache_hit_tokens += token_usage.cache_hit_tokens
            self.total_tokens.cache_miss_tokens += token_usage.cache_miss_tokens

            self.total_cost += task_cost
            self.processed_tasks += 1

            if verbose:
                print(f"  Processed {result_file.parent.name}: ${task_cost.total_cost:.4f}")

    def print_summary(self, successful_tasks: int = None) -> None:
        """Print a summary of costs and statistics"""
        print("\n" + "="*60)
        print("COST CALCULATION SUMMARY")
        print("="*60)
        
        print(f"\nProcessed Tasks: {self.processed_tasks}")
        if self.failed_tasks > 0:
            print(f"Failed/Skipped Tasks: {self.failed_tasks}")
        
        print(f"\nTotal Token Usage:")
        print(f"  Cache Hit Tokens:     {self.total_tokens.cache_hit_tokens:,}")
        print(f"  Cache Miss Tokens:    {self.total_tokens.cache_miss_tokens:,}")
        print(f"  Output Tokens:        {self.total_tokens.total_completion_tokens:,}")
        print(f"  Total Tokens:         {self.total_tokens.total_tokens:,}")

        print(f"\nCost Breakdown:")
        print(f"  Cache Hit Cost:       ${self.total_cost.cache_hit_cost:,.4f}")
        print(f"  Cache Miss Cost:      ${self.total_cost.cache_miss_cost:,.4f}")
        print(f"  Output Cost:          ${self.total_cost.output_cost:,.4f}")
        print(f"  ────────────────────────────────")
        print(f"  TOTAL COST:           ${self.total_cost.total_cost:,.4f}")

        if successful_tasks and successful_tasks > 0:
            avg_cost = self.total_cost.total_cost / successful_tasks
            print(f"\nAverage Cost per Successful Task:")
            print(f"  Total Cost:          ${self.total_cost.total_cost:,.4f}")
            print(f"  Successful Tasks:    {successful_tasks:,}")
            print(f"  Average Cost:        ${avg_cost:.4f} per task")
        
        print("="*60)


def main():
    """Main function - configure directories and settings here"""
    # ============================================================
    # CONFIGURATION - Set your directories and settings here
    # ============================================================
    
    # List of directories to scan for result.json files
    # Each directory should contain subdirectories with result.json files
    directories = [
        # these folders are archived
        "data/converted_alfred/generated_code",
        "data/converted_bddl/generated_code",
        "data/converted_neiss/generated_code",
        "data/converted_normbank/generated_code",
        "data/converted_virtualhome/generated_code",
    ]
    
    # Number of successful tasks for average cost calculation
    successful_tasks = 12280
    
    # Set to True for detailed output per task
    verbose = False
    
    # ============================================================
    # END CONFIGURATION
    # ============================================================
    
    calculator = CostCalculator()

    # Process each directory
    for dir_path_str in directories:
        dir_path = Path(dir_path_str)
        print(f"\nProcessing directory: {dir_path}")
        calculator.scan_directory(dir_path, verbose=verbose)

    # Print summary
    calculator.print_summary(successful_tasks=successful_tasks)


if __name__ == "__main__":
    main()