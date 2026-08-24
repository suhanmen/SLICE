import os
import json
from pathlib import Path

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from .instruction.base import base_prompt
except ImportError:  # Fallback when executed as a standalone script
    import sys
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    if str(PARENT_DIR) not in sys.path:
        sys.path.append(str(PARENT_DIR))
    from instruction.base import base_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
########################################################################

def _read_existing_single(save_path, model_key):
    if not os.path.exists(save_path):
        return {}, set()
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}, set()

    done = set()
    for example in data['output']:
        tid = example.get("task_id")
        done.add(tid)
    return data, done

def save_json(data, path, option_indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent= option_indent)            

def save_jsonl(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in data:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")


def load_dataset(args):
    data_types = ['train', 'eval', 'test']
    data_dict = {data_type: [] for data_type in data_types}
    original_path = DATASET_ROOT / args.dataset
    
    for data_type in data_types:
        path = original_path / f"{data_type}.json"
        data = load_dataset_file(path)
        for idx, example in enumerate(data):
            input, output = build_prompt(example, args.use_instruction)
            data_dict[data_type].append({
                'dataset_name': args.dataset,
                'ID': idx,
                'input': input,
                'label': output,
                'full_data': example
            })   

    return data_dict

def build_prompt(Example, use_instruction):
    if use_instruction == "initial":
        return Example['input'], Example['label']
    if use_instruction == "base":
        return base_prompt.format(instruction=Example['instruction'], input=Example['input']), Example['output']
    else:
        raise ValueError(f"Instruction {use_instruction} not found")
    
def load_output_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_contract_dataset(path):
    """Load the contract tasks.

    Each task is a dict:
        "id":          identifier
        "description": problem / contract text (used to build prompts)
        "signature":   optional "def name(args):" header
        "valid_tests": list of call snippets that MUST run without raising
                       (valid inputs -> functionality preserved)
        "cvts":        list of call snippets that SHOULD raise AssertionError
                       (Contract Violation Tests -> intended rejection)

    Held-out CVTs for final scoring must be disjoint from any used during guidance.
    Returns the list of task dicts (accepts a bare list or {id: task} mapping).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data if isinstance(data, list) else list(data.values())
    for t in tasks:
        t.setdefault("valid_tests", [])
        t.setdefault("valid_checks", [])
        t.setdefault("cvts", [])
    return tasks

def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_jsonl(path):
    path = Path(path)
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_csv(path, **kwargs):
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for CSV loading. Install it with: pip install pandas")
    path = Path(path)
    return pd.read_csv(path, **kwargs)

def load_parquet(path, **kwargs):
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for Parquet loading. Install it with: pip install pandas")
    path = Path(path)
    return pd.read_parquet(path, **kwargs)

def load_dataframe(path, **kwargs):
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for DataFrame loading. Install it with: pip install pandas")
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        return pd.read_csv(path, **kwargs)
    elif suffix == '.parquet':
        return pd.read_parquet(path, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(path, **kwargs)
    elif suffix == '.tsv':
        return pd.read_csv(path, sep='\t', **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .csv, .parquet, .xlsx, .xls, .tsv")

def load_dataset_file(path):
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    suffix = path.suffix.lower()
    
    # Load based on file extension
    if suffix == '.json':
        data = load_json(path)
        return data
    
    elif suffix == '.jsonl':
        data = load_jsonl(path)
        return data
    
    elif suffix == '.csv':
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for CSV loading. Install it with: pip install pandas")
        data = load_csv(path)
    
    elif suffix == '.parquet':
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for Parquet loading. Install it with: pip install pandas")
        data = load_parquet(path)
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return data