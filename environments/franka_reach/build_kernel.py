"""Load the hand-written CUDA step for this task."""
from pathlib import Path
def physics_source(tag):
 return Path(__file__).with_name("task_step.cuh").read_text()
