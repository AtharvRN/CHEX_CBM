#!/usr/bin/env python3
"""
Test script to verify MedGemma pipeline API implementation.

This script tests that the eval_chexagent.py is consistent with
the official MedGemma usage pattern.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Test imports
try:
    from transformers import pipeline
    print("✓ pipeline import works")
except ImportError as e:
    print(f"✗ pipeline import failed: {e}")
    sys.exit(1)

# Check that the script has the right structure
import ast
import inspect

print("\nVerifying eval_chexagent.py structure...")

# Read the file
with open("scripts/eval_chexagent.py") as f:
    code = f.read()

# Check for key components
checks = {
    "pipeline import": "from transformers import pipeline" in code,
    "pipeline_answer function": "def pipeline_answer(" in code,
    "load_hf_pipeline function": "def load_hf_pipeline(" in code,
    "--use_pipeline argument": '"--use_pipeline"' in code,
    "--system_prompt argument": '"--system_prompt"' in code,
    "messages format with system role": '"role": "system"' in code,
    "image-text-to-text task": '"image-text-to-text"' in code,
    "pipe(text=messages)": "pipe(text=messages" in code,
}

all_passed = True
for check_name, result in checks.items():
    status = "✓" if result else "✗"
    print(f"{status} {check_name}")
    if not result:
        all_passed = False

print("\n" + "="*60)
if all_passed:
    print("All checks passed! ✓")
    print("\nThe implementation is now consistent with official MedGemma usage:")
    print("""
    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device="cuda"
    )
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "..."}]},
        {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image", "image": image}
        ]}
    ]
    
    output = pipe(text=messages, max_new_tokens=200)
    """)
    print("\nUsage:")
    print("  python scripts/eval_chexagent.py \\")
    print("    --model_backend hf_vlm \\")
    print("    --model_id google/medgemma-4b-it \\")
    print("    --use_pipeline \\")
    print("    --label_set chexpert \\")
    print("    --data_dir /path/to/CheXpert")
    print("\nNote: --use_pipeline is auto-enabled for MedGemma models")
else:
    print("Some checks failed! ✗")
    sys.exit(1)

print("="*60)
