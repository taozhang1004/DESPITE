import pandas as pd
import json
import time
import openai
import asyncio
from typing import Dict, Any, List, Union
from pathlib import Path
from tqdm import tqdm
import os
from dotenv import load_dotenv

from adaptor import get_adaptor, BaseAdaptor
from danger_extractor import DangerExtractor

class DatasetConverter:
    def __init__(self, api_key: str, dataset_name: str, max_concurrent: int = 10, temperature: float = 0.0):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.dataset_name = dataset_name
        self.adaptor = get_adaptor(dataset_name)
        # Log directory will be set based on output directory later
        self.danger_extractor = DangerExtractor(api_key, max_concurrent, temperature=temperature)
    
    async def process_dataset(self, source_path: str, row_start: int = 0, row_end: int = 100, **filter_kwargs) -> Dict[str, Any]:
        """Process dataset using adaptor and danger extractor"""
        # Set up log directory in the same location as output files
        output_dir = Path(__file__).parent.parent.parent.parent / "data" / f"converted_{self.dataset_name}"
        folder_name = f"tasks-{row_start}-{row_end if row_end != -1 else 'end'}"
        task_dir = output_dir / folder_name
        log_dir = task_dir / "logs"
        
        # Update danger extractor with proper log directory
        from llm_logger import LLMLogger
        self.danger_extractor.logger = LLMLogger(log_dir)
        
        # Load and filter data using adaptor
        df = self.adaptor.load_data(source_path)
        filtered_rows = self.adaptor.filter_data(df, row_start=row_start, row_end=row_end, **filter_kwargs)
        
        print(f"Found {len(filtered_rows)} filtered rows in range {row_start}-{row_end if row_end != -1 else 'end'}")
        
        # Create tasks for concurrent processing
        tasks = []
        for idx, row in filtered_rows.iterrows():
            # Use adaptor to create data string and extract metadata
            data_string = self.adaptor.create_data_string(row)
            metadata = self.adaptor.extract_metadata(row)
            
            # Safely convert index to int
            row_idx = int(idx) if isinstance(idx, (int, float)) else 0
            task = self.danger_extractor.extract_danger_async(data_string, row_idx, metadata)
            tasks.append(task)
        
        # Warm up cache with first task if any tasks exist
        if tasks:
            print("Warming up cache with first task...")
            first_result = await tasks[0]
            remaining_tasks = tasks[1:]
            # Wait a bit for cache to be constructed
            print("Waiting for cache construction...")
            await asyncio.sleep(5)  # Give cache 5 seconds to build
        else:
            first_result = None
            remaining_tasks = []
        
        # Process remaining tasks concurrently with progress bar
        danger_tasks = []
        filtered_out_tasks = []
        
        # Add first result if it exists
        if first_result is not None:
            if isinstance(first_result, Exception):
                print(f"Error in cache warming: {first_result}")
            else:
                # Process result using danger extractor methods
                if self.danger_extractor.is_danger_extracted(first_result):
                    # Add reviewer_ids field for verification tracking
                    first_result['reviewer_ids'] = []
                    danger_tasks.append(first_result)
                else:
                    filtered_out_tasks.append(first_result)
        
        # Create progress bar with callback pattern
        total_tasks = len(tasks)
        with tqdm(total=total_tasks, desc="Extracting danger information", initial=1 if first_result is not None else 0) as pbar:
            # Progress callback function
            async def progress_callback(result):
                nonlocal danger_tasks, filtered_out_tasks
                
                if isinstance(result, Exception):
                    print(f"Error: {result}")
                    pbar.update(1)
                    return
                
                # Process result using danger extractor methods
                if self.danger_extractor.is_danger_extracted(result):
                    # Add reviewer_ids field for verification tracking
                    result['reviewer_ids'] = []
                    danger_tasks.append(result)
                else:
                    filtered_out_tasks.append(result)
                
                pbar.set_postfix({'Danger Extracted': len(danger_tasks), 'Filtered Out': len(filtered_out_tasks)})
                pbar.update(1)
            
            # Create tasks with progress callback
            callback_tasks = []
            for task in remaining_tasks:
                # Wrap the task to call the callback when done
                async def wrapped_task(task, callback):
                    try:
                        result = await task
                        await callback(result)
                        return result
                    except Exception as e:
                        await callback(e)
                        raise e
                
                callback_tasks.append(wrapped_task(task, progress_callback))
            
            # Run tasks concurrently
            if callback_tasks:
                await asyncio.gather(*callback_tasks, return_exceptions=True)
        
        # Save results
        danger_file = self._save_danger_tasks(danger_tasks, row_start, row_end)
        filtered_file = self._save_filtered_tasks(filtered_out_tasks, row_start, row_end)
        
        return {
            'total_processed': len(filtered_rows),
            'danger_extracted_count': len(danger_tasks),
            'filtered_out_count': len(filtered_out_tasks),
            'danger_file': danger_file,
            'filtered_file': filtered_file
        }



    
    def _save_danger_tasks(self, tasks: List[Dict], row_start: int, row_end: int) -> str:
        """Save tasks with extracted danger information to JSON file"""
        output_data = {
            'metadata': {
                'total_tasks': len(tasks),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'row_range': f"{row_start}-{row_end if row_end != -1 else 'end'}",
                'dataset': self.dataset_name
            },
            'danger_tasks': tasks
        }
        
        output_dir = Path(__file__).parent.parent.parent.parent / "data" / f"converted_{self.dataset_name}"
        output_dir.mkdir(exist_ok=True)
        
        # Create folder with row range naming
        folder_name = f"tasks-{row_start}-{row_end if row_end != -1 else 'end'}"
        task_dir = output_dir / folder_name
        task_dir.mkdir(exist_ok=True)
        
        output_file = task_dir / "danger_extracted_tasks.json"
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return str(output_file)
    
    def _save_filtered_tasks(self, tasks: List[Dict], row_start: int, row_end: int) -> str:
        """Save filtered out tasks to JSON file"""
        output_data = {
            'metadata': {
                'total_tasks': len(tasks),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'row_range': f"{row_start}-{row_end if row_end != -1 else 'end'}",
                'dataset': self.dataset_name
            },
            'filtered_tasks': tasks
        }
        
        output_dir = Path(__file__).parent.parent.parent.parent / "data" / f"converted_{self.dataset_name}"
        output_dir.mkdir(exist_ok=True)
        
        # Create folder with row range naming
        folder_name = f"tasks-{row_start}-{row_end if row_end != -1 else 'end'}"
        task_dir = output_dir / folder_name
        task_dir.mkdir(exist_ok=True)
        
        output_file = task_dir / "filtered_out_tasks.json"
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return str(output_file) 


async def main():
    """Main function to run dataset conversion pipeline"""
    import os
    import sys
    from pathlib import Path
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Configuration parameters
    DATASET_NAME = "normbank"  # Change this for different datasets
    ROW_START = 110000 # Start row index
    ROW_END = 112000 # End row index (or -1 for all rows from start)
    MAX_TASKS = -1  # Use -1 for all tasks
    MAX_CONCURRENT = 10
    TEMPERATURE = 1.5  # LLM temperature (0.0 = deterministic, higher = more creative)
    
    # Get operation from command line argument
    if len(sys.argv) < 2:
        print("Usage: python converter.py <operation>")
        print("Operations: filter")
        return
    
    operation = sys.argv[1]
    if operation not in ['filter']:
        print("Invalid operation. Use: filter")
        return
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Set DEEPSEEK_API_KEY environment variable")
        return
    
    # Dataset-agnostic source path - different datasets use different files/folders
    source_files = {
        "normbank": "NormBank.csv",
        "neiss": "neiss2024.csv",
        "bddl": "bddl"  # BDDL uses directory structure
    }
    
    if DATASET_NAME not in source_files:
        print(f"Unknown dataset: {DATASET_NAME}. Available: {list(source_files.keys())}")
        return
    
    source_path = Path("data") / "source" / source_files[DATASET_NAME]
    converter = DatasetConverter(api_key, DATASET_NAME, max_concurrent=MAX_CONCURRENT, temperature=TEMPERATURE)
    
    if operation == 'filter':
        print(f"Processing {DATASET_NAME} dataset with danger extraction...")
        print(f"Using temperature: {TEMPERATURE}")
        # Pass dataset-specific parameters to adaptor via **kwargs
        if DATASET_NAME == "normbank":
            results = await converter.process_dataset(str(source_path), row_start=ROW_START, row_end=ROW_END, filter_norm="taboo")
        else:
            # Other datasets can have their own parameters
            results = await converter.process_dataset(str(source_path), row_start=ROW_START, row_end=ROW_END)
        
        print(f"\nDanger Extraction Results:")
        print(f"  Danger Extracted: {results['danger_extracted_count']} tasks")
        print(f"  Filtered Out: {results['filtered_out_count']} tasks")
        print(f"  Danger tasks saved to: {results['danger_file']}")
        print(f"  Filtered tasks saved to: {results['filtered_file']}")
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())