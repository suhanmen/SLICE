base_prompt = """
You are an expert C++ code style transformer.

I will provide:
- Instruction: the description of the desired style changes.
- Input Code (C++): the code to be transformed.

Requirements:
- Preserve the original program behavior.
- Apply only stylistic changes according to the Instruction.
- Do NOT change algorithms or logic unless explicitly requested.

Output:
- Return ONLY one JSON code block with language `cpp`.
- Do NOT include any explanation or text outside the code block.
- Generate ONLY ONE final answer; do NOT output multiple candidate solutions or multiple JSON blocks.

### Output Example
```json
{{ 
  "code": '''
  ...
  '''
}}
```
----------------------------------------
Instruction: {instruction}
Input Code:
{input}

Output:
"""