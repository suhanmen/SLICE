"""Always persist a scoring or analysis result to a file.

Saving is the default, not a flag: `--save <path>` only changes where the file goes, and the path
is printed as the last line of stdout. A table that exists only in a terminal log cannot be
verified or recovered later without recomputing it.

Default location: evaluation/scoring/<script>_<YYYYmmdd-HHMM>.json
"""

import json
import os
import sys
import time

# scripts/persist.py -> the parent directory is the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(BASE_DIR, "evaluation", "scoring")


def take_save_flag(argv):
    """Strip `--save <path>` out of argv. Returns (remaining argv, path or None).

    A script that already uses argparse can add the argument to its own parser instead.
    """
    if "--save" not in argv:
        return list(argv), None
    i = argv.index("--save")
    path = argv[i + 1] if i + 1 < len(argv) else None
    return list(argv[:i]) + list(argv[i + 2:]), path


def save(script_name, payload, override=None, quiet=False):
    """Write the result dict as JSON and return the path.

    With `override`, that path is used; otherwise a timestamped name in the default directory.
    A write failure must not take the scoring run down with it, so it only warns.
    """
    try:
        if override:
            path = override
            d = os.path.dirname(os.path.abspath(path))
        else:
            d = DEFAULT_DIR
            stamp = time.strftime("%Y%m%d-%H%M")
            path = os.path.join(d, f"{script_name}_{stamp}.json")
        os.makedirs(d, exist_ok=True)
        payload = dict(payload)
        payload.setdefault("_meta", {})
        payload["_meta"].update({
            "script": script_name,
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        with open(path, "w") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
        if not quiet:
            print(f"saved: {path}")
        return path
    except Exception as exc:          # never lose the scores over a failed write
        print(f"[warn] could not save the result ({exc}); console output only", file=sys.stderr)
        return None
