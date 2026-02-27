import numpy as np, os
d = 'Simulation/grid_map/'
for f in sorted(os.listdir(d)):
    if f.endswith('.npy'):
        arr = np.load(d + f)
        print(f'{f:<45} {arr.shape}')
