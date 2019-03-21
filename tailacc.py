import torch
import numpy as np

results = torch.load('val_set_results3.pt')
targets = torch.load('val_targets.pt')

results = results[:,1:]
results = results.astype(np.float)
targets = np.array(targets)

res_target = np.concatenate((results, targets), axis=1)

def tailacc(results, threshold):
    correct = np.zeros(20)
    total = (results[:,:20] > threshold).sum(axis=0)

    for idx, row in enumerate(results):
        pred = (row[:20] > threshold)
        row_correct = pred * row[20:]
        correct += row_correct
    return correct/total
# def tailacc(results, threshold, cls, uppertail):
#     top_n =len(results) - int(uppertail / 100 * len(results))

#     results = results[results[:,cls].argsort()][top_n:]
#     correct = 0
#     total = (results[:,cls] > threshold).sum(axis=0)

#     for idx, row in enumerate(results):
#         pred = (row[cls] > threshold)
       
#         row_correct = pred * row[20+cls]
#        # print(row_correct)
#         correct += row_correct
#     return correct/total

x = np.linspace(0,1,num=20, endpoint=False)
arr = []
for i in x:
    arr.append(tailacc(res_target, threshold=i).sum()/20 * 100)
    #print(tailacc(res_target, threshold=i,cls=1, uppertail=25) * 100)
    print(tailacc(res_target, threshold=i).sum()/20 * 100)

#print(tailacc(res_target, 0.5, 0, 25))
import matplotlib.pyplot as plt

plt.plot(x, arr)
plt.xlabel("threshold")
plt.ylabel("Mean tail accuracy")
plt.title('Tail accuracy against threshold')
# plt.savefig(save +'_2.png')
plt.show()