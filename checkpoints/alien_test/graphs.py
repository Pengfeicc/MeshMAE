import matplotlib.pyplot as plt

# read log.txt file
file_path = 'log.txt'
epochs = []
train_losses = []
train_accs = []
valid_accs = []

with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if 'epoch' in line:
            parts = line.split()
            epochs.append(int(parts[1]))
            train_losses.append(float(parts[4]))
            train_accs.append(float(parts[7]))
            valid_accs.append(float(parts[10]))


# Plotting training accuracy and validation accuracy per epoch
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accs, label='Train Accuracy', marker='o')
plt.plot(epochs, valid_accs, label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Plotting training loss as epoch changes
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()
