from matplotlib import pyplot as plt
import torch

results_A = torch.load('./resnet34_A_results.pt')
results_B = torch.load('./resnet34_B_results.pt')
results_C = torch.load('./resnet34_C_results.pt')

#plt.subplot(3,1,1)
plt.plot(range(30), results_A['train_loss'][:30], label='mode A')
plt.plot(range(15), results_B['train_loss'][:15], label='mode B')
plt.plot(range(15), results_C['train_loss'][:15], label='mode C')
plt.xlabel("No. of Epochs")
plt.ylabel("Training loss")
plt.legend()
plt.savefig('ABC_1.png')
plt.show()

plt.plot(range(30), results_A['val_loss'][:30], label='mode A')
plt.plot(range(15), results_B['val_loss'][:15], label='mode B')
plt.plot(range(15), results_C['val_loss'][:15], label='mode C')
plt.xlabel("No. of Epochs")
plt.ylabel("Validation loss")
plt.legend()
plt.savefig('ABC_2.png')
plt.show()

plt.plot(range(30), results_A['val_acc'][:30], label='mode A')
plt.plot(range(15), results_B['val_acc'][:15], label='mode B')
plt.plot(range(15), results_C['val_acc'][:15], label='mode C')
plt.xlabel("No. of Epochs")
plt.ylabel("Validation accuracy")
plt.legend()
plt.savefig('ABC_3.png')
plt.show()



