"""
LLM logging utility for tracking detailed token usage and responses
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

class LLMLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create separate log files
        self.danger_extraction_log = self.log_dir / "danger_extraction.jsonl"
        self.action_planning_log = self.log_dir / "action_planning.jsonl"
    
    def log_danger_extraction(self, idx: int, metadata: Dict[str, Any], 
                            llm_response: str, usage_stats: Dict[str, Any],
                            processing_time: float):
        """Log danger extraction LLM call details"""
        log_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'idx': idx,
            'metadata': metadata,
            'llm_response': llm_response,
            'token_usage': {
                'prompt_tokens': usage_stats.get('prompt_tokens', 0),
                'completion_tokens': usage_stats.get('completion_tokens', 0),
                'total_tokens': usage_stats.get('total_tokens', 0),
                'prompt_cache_hit_tokens': usage_stats.get('prompt_cache_hit_tokens', 0),
                'prompt_cache_miss_tokens': usage_stats.get('prompt_cache_miss_tokens', 0)
            },
            'processing_time_seconds': processing_time,
            'cache_hit_rate': self._calculate_cache_hit_rate(usage_stats)
        }
        
        self._append_to_log(self.danger_extraction_log, log_entry)
    
    def log_action_planning(self, task_info: Dict[str, Any], 
                          llm_response: str, usage_stats: Dict[str, Any],
                          processing_time: float):
        """Log action planning LLM call details"""
        log_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'task_idx': task_info.get('idx'),
            'task_metadata': task_info.get('metadata'),
            'llm_response': llm_response,
            'token_usage': {
                'prompt_tokens': usage_stats.get('prompt_tokens', 0),
                'completion_tokens': usage_stats.get('completion_tokens', 0),
                'total_tokens': usage_stats.get('total_tokens', 0),
                'prompt_cache_hit_tokens': usage_stats.get('prompt_cache_hit_tokens', 0),
                'prompt_cache_miss_tokens': usage_stats.get('prompt_cache_miss_tokens', 0)
            },
            'processing_time_seconds': processing_time,
            'cache_hit_rate': self._calculate_cache_hit_rate(usage_stats)
        }
        
        self._append_to_log(self.action_planning_log, log_entry)
    
    def _calculate_cache_hit_rate(self, usage_stats: Dict[str, Any]) -> float:
        """Calculate cache hit rate percentage"""
        cache_hit_tokens = usage_stats.get('prompt_cache_hit_tokens', 0)
        total_tokens = usage_stats.get('prompt_tokens', 0)
        return (cache_hit_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    def _append_to_log(self, log_file: Path, entry: Dict[str, Any]):
        """Append a single JSON entry to log file"""
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from all log files"""
        stats = {
            'danger_extraction': self._get_file_stats(self.danger_extraction_log),
            'action_planning': self._get_file_stats(self.action_planning_log)
        }
        return stats
    
    def _get_file_stats(self, log_file: Path) -> Dict[str, Any]:
        """Get statistics from a specific log file"""
        if not log_file.exists():
            return {'total_calls': 0, 'total_tokens': 0, 'avg_cache_hit_rate': 0}
        
        total_calls = 0
        total_tokens = 0
        total_cache_hit_rate = 0
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    total_calls += 1
                    total_tokens += entry.get('token_usage', {}).get('total_tokens', 0)
                    total_cache_hit_rate += entry.get('cache_hit_rate', 0)
                except json.JSONDecodeError:
                    continue
        
        return {
            'total_calls': total_calls,
            'total_tokens': total_tokens,
            'avg_cache_hit_rate': total_cache_hit_rate / total_calls if total_calls > 0 else 0
        }