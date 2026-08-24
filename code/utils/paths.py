"""One rule for where every generated file goes: output/<setting>/<model>/<name>.

`setting` names the configuration being run (`SLICE` for the method, `wograph`, `base_cs`, ...),
so a second configuration on the same model never overwrites the first. `final.json` is the file
each setting is scored on, which is what results_store.py reads.

    conditions.json      (i)   contract conditions extracted from the description
    view.json            (i)   functional view (the masked description)
    bodies_greedy.json   (ii)  greedy candidate
    bodies_sampled.json  (ii)  temperature candidates
    selected.json        (ii)  the chosen body per task
    assertions_raw.json  (iii) model output before post-processing
    inserted.json        (iii) after AST insertion
    final.json           (iii) after screening — the scored output

Dataset-derived files that no model influences stay in artifacts/ and are read-only.
"""
import os

DEFAULT_SETTING = "SLICE"


def run_dir(setting, model, create=True):
    d = os.path.join("output", setting, model)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def run_path(setting, model, name, create=True):
    """Path of one generated file; `name` may be given with or without the .json suffix."""
    if not name.endswith(".json"):
        name += ".json"
    return os.path.join(run_dir(setting, model, create), name)
