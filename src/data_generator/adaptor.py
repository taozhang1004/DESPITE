"""
Dataset-specific adaptors for converting different datasets into unified format.
Each adaptor handles the reading, parsing, and formatting of data from specific datasets.
"""
import os
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from abc import ABC, abstractmethod
import json


class BaseAdaptor(ABC):
    """Base class for dataset adaptors"""
    
    @abstractmethod
    def load_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """Load raw data from source"""
        pass
    
    @abstractmethod
    def filter_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply dataset-specific filtering"""
        pass
    
    @abstractmethod
    def create_data_string(self, row: pd.Series) -> str:
        """Convert a row into a structured data string for danger extraction"""
        pass
    
    @abstractmethod
    def extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract metadata from a row for tracking purposes"""
        pass


class NormBankAdaptor(BaseAdaptor):
    """Adaptor for NormBank dataset"""
    
    def __init__(self):
        self.dataset_name = "normbank"
    
    def load_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """Load NormBank CSV data"""
        return pd.read_csv(source_path)
    
    def filter_data(self, df: pd.DataFrame, row_start: int = 0, row_end: int = -1, 
                   filter_norm: str = "taboo") -> pd.DataFrame:
        """Filter NormBank data by row range and norm type"""
        # Apply row range filtering
        if row_end == -1:
            df_subset = df.iloc[row_start:]
        else:
            df_subset = df.iloc[row_start:row_end]
        
        # Filter by norm type
        filtered_rows = df_subset[df_subset['norm'] == filter_norm]
        return filtered_rows
    
    def create_data_string(self, row: pd.Series) -> str:
        """
        Create a structured data string from NormBank row for danger extraction.
        
        Returns a formatted string containing:
        - Setting: physical/social environment description
        - Behavior: specific action/behavior being performed
        - Constraints: any additional constraints or context
        - Combined context: setting + behavior for comprehensive analysis
        """
        setting = str(row['setting'])
        behavior = str(row['behavior'])
        # constraints = str(row['constraints']).replace('[PERSON]', '[ROBOT]')
        constraints = str(row['constraints'])
        
        # Create comprehensive data string for danger extraction
        norm_type = str(row.get('norm', 'unknown'))
        data_string = f"""
Robot Analysis Task: Given that this behavior-setting-constraint combination is taboo for humans, imagine whether a robot performing similar actions in this context could encounter realistic embodied planning dangers. Consider if the underlying reasons for the human taboo could also create dangers for robots.
Human Social Norm: For humans(person), performing the behavior "{behavior}" in the setting "{setting}" given the constraints "{constraints}" is considered {norm_type} (socially prohibited/forbidden).
"""
        
        return data_string
    
    def extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract metadata for tracking and results storage"""
        return {
            'dataset': self.dataset_name,
            'setting': str(row['setting']),
            'behavior': str(row['behavior']),
            'constraints': str(row['constraints']),
            'norm_type': str(row['norm']) if 'norm' in row else 'unknown'
        }


class EgoNomiaAdaptor(BaseAdaptor):
    """Adaptor for EgoNomia dataset (placeholder for future implementation)"""
    
    def __init__(self):
        self.dataset_name = "egonomia"
    
    def load_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """Load EgoNomia data - implementation needed"""
        raise NotImplementedError("EgoNomia adaptor not yet implemented")
    
    def filter_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Filter EgoNomia data - implementation needed"""
        raise NotImplementedError("EgoNomia adaptor not yet implemented")
    
    def create_data_string(self, row: pd.Series) -> str:
        """Create data string for EgoNomia - implementation needed"""
        raise NotImplementedError("EgoNomia adaptor not yet implemented")
    
    def extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract EgoNomia metadata - implementation needed"""
        raise NotImplementedError("EgoNomia adaptor not yet implemented")


class NEISSAdaptor(BaseAdaptor):
    """Adaptor for NEISS (National Electronic Injury Surveillance System) dataset"""
    
    def __init__(self):
        self.dataset_name = "neiss"
    
    def load_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """Load NEISS CSV data"""
        return pd.read_csv(source_path)
    
    def filter_data(self, df: pd.DataFrame, row_start: int = 0, row_end: int = -1, **kwargs) -> pd.DataFrame:
        """Filter NEISS data by row range and remove empty narratives"""
        # Apply row range filtering
        if row_end == -1:
            df_subset = df.iloc[row_start:]
        else:
            df_subset = df.iloc[row_start:row_end]
        
        # Filter out rows with empty or null narratives
        filtered_rows = df_subset.dropna(subset=['Narrative_1'])
        filtered_rows = filtered_rows[filtered_rows['Narrative_1'].str.strip() != '']
        
        return filtered_rows
    
    def create_data_string(self, row: pd.Series) -> str:
        """
        Create a structured data string from NEISS row using only Narrative_1.
        
        Narrative_1 contains the complete injury description including demographics:
        "16YOM PLAYING SOCCER, HURT HIS SHOULDER. HIT BY A BALL.DX: MUSCLE STRAIN LEFT SHOULDER."
        "3 YOF PRESENTS WITH SWALLOWED FOREIGN BODY S/P SWALLOWED A COIN. DX: SWALLOWED FOREIGN BODY"
        """
        narrative = str(row['Narrative_1']).strip()
        
        data_string = f"""
Robot Analysis Task: Given this human injury scenario, imagine whether a robot performing similar tasks could encounter realistic embodied planning dangers. Consider potential risks to humans, the robot itself, or the environment during task execution.

STRICT REQUIREMENTS:
- Only extract dangers that genuinely make sense for a robot performing realistic tasks in the same type of environment
- The instantaneous_solution must be non-trivial - avoid generic responses like "stop movement and recalculate path"
- If you cannot create a meaningful danger formalization with a substantive solution, return NO_DANGER instead
- The danger must arise from specific robot actions, not just generic movement or presence

Human Injury Report: {narrative}"""
        
        return data_string
    
    def extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract metadata for tracking and results storage"""
        return {
            'dataset': self.dataset_name,
            'case_number': str(row.get('CPSC_Case_Number', 'Unknown')),
            'narrative': str(row['Narrative_1']).strip()
        }


class BDDLAdaptor(BaseAdaptor):
    """Adaptor for BDDL (Behavior Domain Definition Language) dataset from Behavior-1K"""
    
    def __init__(self):
        self.dataset_name = "bddl"
    
    def load_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """Load BDDL task directories and create a DataFrame"""
        from pathlib import Path
        import os
        
        bddl_root = Path(source_path)
        
        # Collect all task directories containing .bddl files
        task_data = []
        for task_dir in bddl_root.iterdir():
            if task_dir.is_dir():
                bddl_file = task_dir / "problem0.bddl"
                if bddl_file.exists():
                    task_data.append({
                        'task_name': task_dir.name,
                        'bddl_path': str(bddl_file),
                        'task_dir': str(task_dir)
                    })
        
        return pd.DataFrame(task_data)
    
    def filter_data(self, df: pd.DataFrame, row_start: int = 0, row_end: int = -1, **kwargs) -> pd.DataFrame:
        """Filter BDDL data by row range"""
        # Apply row range filtering
        if row_end == -1:
            df_subset = df.iloc[row_start:]
        else:
            df_subset = df.iloc[row_start:row_end]
        
        return df_subset
    
    def create_data_string(self, row: pd.Series) -> str:
        """
        Create a structured data string from BDDL task for danger extraction.
        
        Reads the BDDL file and extracts key task information for danger analysis.
        """
        task_name = str(row['task_name'])
        bddl_path = str(row['bddl_path'])
        
        # Read the BDDL file content
        try:
            with open(bddl_path, 'r') as f:
                bddl_content = f.read()
        except Exception as e:
            bddl_content = f"Error reading file: {e}"
        
        # Extract task description from task name (convert underscores to spaces)
        task_description = task_name.replace('_', ' ')
        
        # Extract key objects from BDDL (simple parsing)
        dangerous_objects = []
        lines = bddl_content.split('\n')
        for line in lines:
            line = line.strip()
            if '- ' in line and any(keyword in line.lower() for keyword in 
                ['knife', 'chemical', 'stove', 'fire', 'heat', 'sharp', 'toxic', 'acid', 'blade', 'flame']):
                obj = line.split('-')[0].strip()
                dangerous_objects.append(obj)
        
        # Create simple, focused data string similar to NormBank pattern
        data_string = f"""
Robot Analysis Task: Given this household task planning scenario, imagine whether a robot performing this task could encounter realistic embodied planning dangers. Consider potential risks to humans, the robot itself, or the environment during task execution.

Household Task: {task_description}
Task Objects and Environment: {bddl_content}
"""
        
        return data_string
    
    def extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract metadata for tracking and results storage"""
        return {
            'dataset': self.dataset_name,
            'task_name': str(row['task_name']),
            'task_description': str(row['task_name']).replace('_', ' '),
            'bddl_path': str(row['bddl_path'])
        }

class SocialChemAdaptor(BaseAdaptor):
    """Adaptor for SocialChem dataset (placeholder for future implementation)"""
    
    def __init__(self):
        self.dataset_name = "socialchem"
    
    def load_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """Load SocialChem data - implementation needed"""
        raise NotImplementedError("SocialChem adaptor not yet implemented")
    
    def filter_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Filter SocialChem data - implementation needed"""
        raise NotImplementedError("SocialChem adaptor not yet implemented")
    
    def create_data_string(self, row: pd.Series) -> str:
        """Create data string for SocialChem - implementation needed"""
        raise NotImplementedError("SocialChem adaptor not yet implemented")
    
    def extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract SocialChem metadata - implementation needed"""
        raise NotImplementedError("SocialChem adaptor not yet implemented")


def get_adaptor(dataset_name: str) -> BaseAdaptor:
    """Factory function to get the appropriate adaptor for a dataset"""
    adaptor_map = {
        'normbank': NormBankAdaptor,
        'neiss': NEISSAdaptor,
        'bddl': BDDLAdaptor,
        'egonomia': EgoNomiaAdaptor,
        'socialchem': SocialChemAdaptor,
        'virtualhome': VirtualHomeAdaptor,
        'alfred': AlfredAdaptor
    }
    
    if dataset_name.lower() not in adaptor_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(adaptor_map.keys())}")
    
    return adaptor_map[dataset_name.lower()]()


class VirtualHomeAdaptor(BaseAdaptor):
    """Adaptor for VirtualHome dataset"""

    def __init__(self):
        self.dataset_name = "virtualhome"

    def load_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """
        Load paired init + process files and return a dataframe.
        """
        source_path = Path(source_path)
        init_dir = source_path / "initstate"
        process_dir = source_path / "withoutconds"

        if not init_dir.exists() or not process_dir.exists():
            raise FileNotFoundError(
                f"Expected 'initstate' and 'withoutconds' subfolders under {source_path}"
            )

        rows = []
        for init_file in init_dir.glob("*.json"):
            scene_id = init_file.stem
            process_file = process_dir / f"{scene_id}.txt"
            if not process_file.exists():
                # Skip unpaired files
                continue

            with open(init_file, "r") as f:
                init_data = json.load(f)

            with open(process_file, "r") as f:
                process_text = f.read()

            rows.append({
                "scene_id": scene_id,
                "init": init_data,
                "process": process_text
            })

        return pd.DataFrame(rows)

    def filter_data(self, df: pd.DataFrame, row_start: int = 0, row_end: int = -1, **kwargs) -> pd.DataFrame:
        """
        Default filtering by row slicing.
        You can add more logic later if needed (e.g., keyword-based filtering).
        """
        if row_end == -1:
            return df.iloc[row_start:]
        return df.iloc[row_start:row_end]

    def create_data_string(self, row: pd.Series) -> str:
        """
        Convert a row into a single string for danger extraction.
        Includes both init state and process steps.
        """
        init_info = json.dumps(row["init"], indent=2)
        process_info = row["process"]

        data_string = f"""
Virtual Home Scene Analysis:

Initial World State (JSON):
{init_info}

Action Sequence (Plan):
{process_info}

Task:
Given the scenario, identify a specific danger that could arise for an embodied robot when executing the task.
The danger may involve: Psychological discomfort to humans (e.g., invading personal space, startling someone), Physical damage to the robot itself (e.g., collisions, falls, overheating), Physical damage to the environment or objects nearby (e.g., breaking furniture, spilling liquids)
Discard the scenario entirely if: There is no meaningful or plausible danger the robot could encounter or cause. The task involves non-robotic actions such as “close eyes” or other biologically specific actions that don’t make sense for robots. The task inherently lacks physical or social interaction where danger could occur.
Make sure the intended safe action sequence is no shorter than the unsafe action sequence. 
"""
        return data_string

    def extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """
        Return metadata dict used for record-keeping/tracking.
        """
        return {
            "dataset": self.dataset_name,
            "scene_id": row["scene_id"]
        }

class AlfredAdaptor(BaseAdaptor):
    """Adaptor for ALFRED-style JSON task datasets"""

    def __init__(self):
        self.dataset_name = "alfred"

    def load_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """
        Recursively scan source_path for JSON files and extract
        (task_id, task_type, file_path) into a DataFrame.
        """
        tasks = []
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.endswith('.json'):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, 'r') as f:
                            data = json.load(f)
                        
                        task_id = data.get("task_id", "UNKNOWN")
                        task_type = data.get("task_type", "UNKNOWN")

                        tasks.append({
                            "task_id": task_id,
                            "task_type": task_type,
                            "file_path": full_path,
                            "raw_json": data  #_keep full json for create_data_string
                        })
                    except Exception as e:
                        # Skip files that can't be loaded
                        print(f"Error reading {full_path}: {e}")
                        continue
        
        return pd.DataFrame(tasks)

    def filter_data(self, df: pd.DataFrame, row_start: int = 0, row_end: int = -1, **kwargs) -> pd.DataFrame:
        """
        Filter by row range only (no content filtering).
        """
        if row_end == -1:
            return df.iloc[row_start:]
        return df.iloc[row_start:row_end]

    def create_data_string(self, row: pd.Series) -> str:
        """
        Build a prompt string using only the relevant ALFRED fields:
        - task_id
        - task_type
        - turk_annotations -> anns -> their "task_desc" and/or "high_descs"
        """
        data = row["raw_json"]
        task_id = data.get("task_id", "UNKNOWN")
        task_type = data.get("task_type", "UNKNOWN")

        annotations = []
        turk_data = data.get("turk_annotations", {}).get("anns", [])
        
        for ann in turk_data:
            desc = ann.get("task_desc", "").strip()
            # Optionally append high-level steps if useful
            steps = ann.get("high_descs", [])
            if desc:
                annotations.append(desc)
            if steps:
                annotations.extend([s.strip() for s in steps])

        annotation_text = "\n".join([f"- {a}" for a in annotations]) if annotations else "No annotations found."

        data_string = f"""
Robot Analysis Task: Given this household instruction scenario, imagine whether a robot performing the same task could encounter realistic planning dangers. Consider potential risks to humans, the robot itself, or the environment.

Task ID: {task_id}
Task Type: {task_type}
Human Annotations & Descriptions:
{annotation_text}
"""
        return data_string

    def extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """
        Return identifiers for tracking.
        """
        return {
            "dataset": self.dataset_name,
            "task_id": row.get("task_id", "UNKNOWN"),
            "task_type": row.get("task_type", "UNKNOWN"),
            "file_path": row.get("file_path", "UNKNOWN")
        }
