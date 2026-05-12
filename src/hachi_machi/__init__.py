import os
import sys

# we want to suppress UserWarning for numpy not being a dependency
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
try:
    import torch
finally:
    sys.stderr = stderr
