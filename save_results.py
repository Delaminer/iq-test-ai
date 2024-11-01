import os
import matplotlib.pyplot as plt
import numpy as np
# from tqdm.notebook import tqdm
models = [('Resnet18_MLP', 'Resnet18_MLP_0.txt'), ('CNN_LSTM', 'CNN_LSTM_0.txt'), ('CNN_MLP', 'CNN_MLP_0.txt')]
folder = 'RAVEN/model_results/IRAVEN-1000'
for model_name, model_file in models:
    with open(os.path.join(folder, model_file), 'r', buffering=1) as f:
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        test_accs = []
        epoch = 0
        epochs = []
        epoch_train_accs = []
        lowest_val_loss = float('inf')
        lowest_val_loss_epoch = 0
        for line in f:
        # for line in tqdm(f):
            if 'Train: Epoch:' in line:
                new_epoch = int(line.split(',')[0].split(':')[-1])
                if new_epoch != epoch:
                    epoch = new_epoch
                    epochs.append(epoch)
                    avg_train_acc = sum(epoch_train_accs) / len(epoch_train_accs)
                    train_accs.append(avg_train_acc)
                    epoch_train_accs = []
                # train_loss = float(line.split(',')[2].split(':')[-1])
                train_acc = float(line.split(':')[-1][:-2])
                epoch_train_accs.append(train_acc)
            if 'Avg Training Loss:' in line:
                train_loss = float(line.split(' ')[-1])
                train_losses.append(train_loss)
            if 'Total Validation Loss:' in line:
                val_loss = float(line.split(' ')[3][:-1])
                val_losses.append(val_loss)
                val_acc = float(line.split(' ')[5])
                val_accs.append(val_acc)
                if val_loss <= lowest_val_loss:
                    lowest_val_loss = val_loss
                    lowest_val_loss_epoch = epoch
            if 'Total Testing Acc:' in line:
                test_acc = float(line.split(' ')[-1])
                test_accs.append(test_acc)

        # Handle last epoch
        avg_train_acc = sum(epoch_train_accs) / len(epoch_train_accs)
        train_accs.append(avg_train_acc)
        epochs.append(epoch)

        shortest_length = min(len(epochs), len(train_losses), len(val_losses), len(train_accs), len(val_accs), len(test_accs))
        epochs = epochs[:shortest_length]
        train_losses = train_losses[:shortest_length]
        val_losses = val_losses[:shortest_length]
        train_accs = train_accs[:shortest_length]
        val_accs = val_accs[:shortest_length]
        test_accs = test_accs[:shortest_length]

        print(f'Lowest validation loss: {lowest_val_loss} at epoch {lowest_val_loss_epoch} with train acc {train_accs[lowest_val_loss_epoch]} and val acc {val_accs[lowest_val_loss_epoch]} and test acc {test_accs[lowest_val_loss_epoch]}')

        # Plot losses
        plt.plot(epochs, train_losses, label='Train')
        plt.plot(epochs, val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Loss')
        plt.legend()

        plt.savefig(os.path.join(folder, f'{model_name}_loss.png'))
        plt.clf()

        # Plot accuracies
        plt.plot(epochs, train_accs, label='Train')
        plt.plot(epochs, val_accs, label='Validation')
        plt.plot(epochs, test_accs, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name} Accuracy')
        plt.legend()


        plt.savefig(os.path.join(folder, f'{model_name}_acc.png'))
        plt.clf()

        print('done with', model_name)