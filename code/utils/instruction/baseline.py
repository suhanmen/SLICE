"""Prompting conditions for the baselines — not used by the method.

Ports ContractEval's own code-generation conditions (its `Instruction.py`:
CODE_GENERATION_CS / CODE_GENERATION_CT, and the contract-test formatting of its
`TG_CG_main.py`) so the baseline numbers stay comparable to the ones reported there:

  base : problem description only, no contract information
  cs   : + the contract preconditions in natural language
  eas  : + the contract-violating inputs, i.e. ContractEval's "Contract Prompt"

These are plain single-pass generations (the model writes the whole function); evaluate
with utils.contract_eval (ContractEval-style per-task strict CSR).

Task schema (from build_contracteval_dataset): {id, entry_point, description (base),
contract_nl, signature, valid_tests, cvts}.
"""

# --- instruction headers, faithful to ContractEval Instruction.py ----------------
_INSTRUCTION = {
    "base": (
        "You are an expert program analyst.\n"
        "I will provide:\n"
        "  - Method Name: the name of the function to implement.\n"
        "  - Problem Description: a one-sentence description of what the function does.\n"
        "Your task is:\n"
        "  1. Read the problem description carefully and infer appropriate arguments.\n"
        "  2. Write a correct implementation that satisfies the described behavior and "
        "passes all provided examples.\n"
        "  3. Do not change the function name."
    ),
    # CODE_GENERATION_CS (Instruction.py:861)
    "cs": (
        "You are an expert program analyst.\n"
        "I will provide:\n"
        "  - Method Name: the name of the function to implement.\n"
        "  - Problem Description: a one-sentence description, followed by one or more "
        "example assertions.\n"
        "  - Contract List: input-validation conditions that must be enforced.\n"
        "Your task is:\n"
        "  1. Read the problem description carefully and infer appropriate arguments.\n"
        "  2. Write a correct implementation that:\n"
        "    - Satisfies the described behavior and passes all provided examples.\n"
        "    - Enforces the input constraints using assert statements.\n"
        "  3. Do not change the function name."
    ),
    # CODE_GENERATION_CT / EAS (Instruction.py:876)
    "eas": (
        "You are an expert program analyst.\n"
        "I will provide:\n"
        "  - Method Name: the name of the function to implement.\n"
        "  - Problem Description: a one-sentence description of what the function does.\n"
        "  - Contract List: assertion-based input validation conditions that must be enforced.\n"
        "  - Contract Test Cases: invalid inputs that must be rejected using appropriate "
        "input validation logic (e.g., assert statements).\n"
        "Your task is:\n"
        "  1. Carefully read the problem description.\n"
        "  2. Analyze the contract test cases to infer input constraints that must be enforced.\n"
        "  3. Write a correct implementation that:\n"
        "    - Passes all functionality examples.\n"
        "    - Rejects all contract test cases by raising an AssertionError.\n"
        "    - Enforces input constraints using assert statements or precondition checks.\n"
        "  4. Do not change the function name."
    ),
}

_OUTPUT_INSTRUCTION = (
    "**Output**\n"
    "- Return ONLY one ```python code block containing the complete function "
    "implementation. Do not include any explanation outside the code block."
)

CONDITIONS = ("base", "cs", "eas")


def _contract_list(task) -> str:
    nl = (task.get("contract_nl") or "").strip()
    return nl if nl else "(none specified)"


def _contract_test_cases(task) -> str:
    """Format the EAS-prompt CVTs exactly like ContractEval (TG_CG_main.py:482):
    `{contract_in_key}:\n>>> entry(invalid_args)\n "AssertionError: invalid input"`.
    Uses `eas_prompt_cvts` (1 CVT per contract clause, DISJOINT from the private_cvts used
    for CSR eval -> no leakage); falls back to `cvts` for older datasets."""
    cvts = task.get("eas_prompt_cvts") or task.get("cvts", [])
    keys = (task.get("eas_prompt_cvt_keys") or task.get("cvt_keys")
            or [f"assert_{i}" for i in range(len(cvts))])
    lines = []
    for call, key in zip(cvts, keys):
        label = key or "assert"
        lines.append(f"{label}:\n>>> {call}\n \"AssertionError: invalid input\"")
    return "\n".join(lines)


def build_baseline_messages(task, condition: str):
    """ContractEval-style code generation (TG_CG_main.code_generation_template_dataset):
    return (system, user) for a chat model. The model generates the COMPLETE function.

    Uses the ORIGINAL ContractEval prompt as the Problem Description (faithful — no
    synthesized signature):
      base : prompt                       (HumanEval: def stub; MBPP: docstring only)
      cs   : prompt_with_contract_aware   (contract described in NL inside the docstring)
      eas  : cs prompt + Contract Test Cases (the violating inputs)
    """
    if condition not in _INSTRUCTION:
        raise ValueError(f"unknown condition {condition!r}; use one of {CONDITIONS}")
    entry = task.get("entry_point") or task.get("id")
    base = (task.get("prompt_base") or "").strip()
    cs = (task.get("prompt_cs") or base).strip()
    desc = cs if condition in ("cs", "eas") else base
    if not desc:   # fallback for non-ContractEval datasets without raw prompts
        sig = task.get("signature") or f"def {entry}(*args):"
        desc = f"{sig}\n    \"\"\"{(task.get('description') or '').strip()}\"\"\""

    system = f"{_INSTRUCTION[condition]}\n\n{_OUTPUT_INSTRUCTION}"
    parts = [f"Method Name: {entry}", f"Problem Description:\n{desc}"]
    if condition == "eas":
        parts.append(f"\nContract Test Cases:\n{_contract_test_cases(task)}")
    return system, "\n".join(parts)
