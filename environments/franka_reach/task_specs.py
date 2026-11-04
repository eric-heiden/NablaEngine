"""Fixed pair lists and baseline capacities for the three audited task models."""
SPECS = {'go2': {'defines': {'NR': 128}, 'pairs': [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [0, 15], [0, 16], [0, 17], [0, 18], [0, 19], [0, 20], [0, 21], [0, 22], [0, 23], [0, 24], [0, 25], [0, 26], [0, 27]]}, 'reach': {'defines': {'NR': 128}, 'pairs': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14]]}, 'lift': {'defines': {'NR': 300}, 'pairs': [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [0, 15], [1, 14], [1, 15], [2, 14], [2, 15], [3, 14], [3, 15], [4, 14], [4, 15], [5, 14], [5, 15], [6, 14], [6, 15], [7, 14], [7, 15], [8, 14], [8, 15], [9, 14], [9, 15], [10, 14], [10, 15], [11, 14], [11, 15], [12, 14], [12, 15], [13, 14], [13, 15], [14, 15]]}}

# Reach cannot touch its lower ground plane within this task's fixed-root kinematic bound.
# Table/base and lift ground/base/table pairs have identically zero dynamic Jacobians.
SPECS['reach']['pairs']=[[1,g] for g in range(3,15)]
SPECS['lift']['pairs']=[p for p in SPECS['lift']['pairs'] if p not in [[0,1],[0,15],[1,15]]]

# Reach retains full collision geometry and its original contact pairs.
SPECS["reach"]["pairs"]=[[a,b] for a in [0,1] for b in range(2,15)]
