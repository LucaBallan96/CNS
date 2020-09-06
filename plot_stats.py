import os
import numpy as np
import matplotlib.pyplot as plt

stats_dir = 'graph_data'
palette = plt.get_cmap('Set1')
plt.style.use('seaborn-darkgrid')
axes = plt.gca()

'''x = ['[5px, 20px]', '[5px, 60px]', '[10px, 90px]', '[20px, 120px]']
y = [93.50, 91.77, 87.17, 86.50]
plt.bar(x, y, width=0.5, color='lightsteelblue')
axes.set_ylim([80.0, 100.0])
axes.set_xlabel('Trigger size range')
axes.set_ylabel('Accuracy (%)')'''


'''x = ['0%', '0.2%', '1%', '5%', '20%']
plt.plot(x, [99.50, 99.33, 99.33, 97.33, 95.67], marker='', color=palette(2), linewidth=1, alpha=0.9, label='full')
plt.plot(x, [99.50, 99.33, 98.83, 96.33, 94.67], marker='', color=palette(3), linewidth=1, alpha=0.9, label='layers 3-4')
plt.plot(x, [98.67, 98.33, 97.83, 94.83, 81.00], marker='', color=palette(4), linewidth=1, alpha=0.9, label='only fc')
plt.legend(loc='lower left')
axes.set_ylim([90.0, 100.0])
axes.set_xlabel('Backdoored images in training')
axes.set_ylabel('Validation accuracy (%)')'''


'''x = ['0%', '0.2%', '1%', '5%', '20%']
plt.plot(x, [99.17, 98.83, 98.67, 99.50, 97.50], marker='', color=palette(2), linewidth=1, alpha=0.9, label='normal test set')
plt.plot(x, [95.00, 94.00, 81.83, 48.67, 9.00], marker='', color=palette(3), linewidth=1, alpha=0.9, label='backdoored test set')
plt.legend()
axes.set_xlabel('Backdoored images in training')
axes.set_ylabel('Accuracy (%)')'''


x = range(30)
for i, trigger_prob in enumerate([0.0, 0.002, 0.01, 0.05, 0.2]):
    model_name = 'dcp_resnet18_layer-full_SGD1e-4_bs8_e30_splittr' + str(trigger_prob) + '_rc256.npy'
    stats = np.load(os.path.join(stats_dir, model_name), allow_pickle=True).item()
    train_accs = np.array(stats['train']) * 100
    val_accs = np.array(stats['val']) * 100
    label = str(trigger_prob * 100) + '%'
    plt.plot(x, train_accs, marker='', color=palette(i), linewidth=1, alpha=0.9, label=label)
plt.legend()
axes.set_xlabel('Training epochs')
axes.set_ylabel('Validation accuracy (%)')

plt.show()
