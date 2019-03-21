from matplotlib import pyplot as plt
import torch

results_A = torch.load('./results/pascalvoc_A_results.pt')

print(max(results_A['val_acc']))

label = 'mode C'
save = 'D'
epoch = 15

plt.plot(range(epoch), results_A['train_loss'][:epoch], label='training loss')

plt.plot(range(epoch), results_A['val_loss'][:epoch], label='val loss')
plt.xlabel("No. of Epochs")
plt.ylabel("Training loss")
plt.legend()
plt.savefig(save +'_1.png')
plt.show()

plt.xlabel("No. of Epochs")
plt.ylabel("Validation loss")

plt.savefig(save +'_2.png')
plt.show()



x = [sum(i)/len(i) for i in results_A['val_acc']]
print(max(x))
plt.plot(range(epoch), x,)
plt.xlabel("No. of Epochs")
plt.ylabel("Validation accuracy")

plt.savefig(save +'_3.png')
plt.show()


