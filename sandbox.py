import torch
import numpy as np
from app.resultviewer import ResultViewer
import os
import os.path
import pandas as pd 
# rv = ResultViewer('D:/Downloads/Deep Learning/Week 6', 'val_set_results3.pt')
# rv.get_class_results('aeroplane')

results = torch.load('val_set_results3.pt')
targets = torch.load('val_targets.pt')

for i in targets:
    print()

# x = torch.load('val_set_results3.pt')
# x = pd.DataFrame(x)
# print(x.loc[:, 1:].reset_index())

# paths = torch.load('val_img_paths.pt')
# val_targets = torch.load('val_targets.pt')
# res = torch.load('val_set_results2.pt')

# paths = [os.path.splitext(os.path.basename(x))[0] for x in paths]
# paths = np.array(paths)

# res = np.array(res)
# print(res.shape, paths.reshape(-1,1).shape)

# x = np.hstack((paths.reshape(-1,1),res))

# i = -1


# print(x[0])
# torch.save(x, 'val_set_results3.pt')