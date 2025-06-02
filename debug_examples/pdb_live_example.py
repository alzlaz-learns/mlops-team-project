# debug_examples/pdb_live_example.py

import pdb


def buggy_function() -> float:
    a = 10
    b = 0
    pdb.set_trace()  # Pauses here
    result = a / b   # Division by zero
    return result

buggy_function()
