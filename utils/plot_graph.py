import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data (using hardcoded data)
iterations = [250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000, 19000, 20000]
train_losses = [6.4552, 5.5645, 4.9581, 4.5593, 4.0041, 3.6584, 3.3993, 3.2888, 3.2119, 3.1809, 3.1208, 3.0619, 2.9755, 2.9221, 2.9747, 2.9280, 2.8887, 2.8001, 2.7417, 2.7102, 2.6675, 2.6611, 2.6514]
val_losses = [6.4574, 5.5790, 4.9899, 4.5965, 4.0609, 3.7355, 3.4914, 3.3781, 3.3047, 3.2752, 3.2080, 3.1709, 3.0928, 3.0334, 3.0985, 3.0523, 3.0114, 2.9345, 2.8935, 2.8692, 2.8337, 2.8102, 2.8118]

sns.set_theme(style="whitegrid", palette="muted")
plt.figure(figsize=(15, 8))

plt.plot(iterations, train_losses, 'o-', label='Train Loss', color='royalblue')
plt.plot(iterations, val_losses, 'o-', label='Validation Loss', color='orangered')

best_val_loss = 2.8102
best_iter = 19000
plt.scatter(best_iter, best_val_loss, s=200, color='gold', edgecolor='black', zorder=5, label='Best PPL: 16.61')
plt.title('Training Convergence', fontsize=20)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.legend()
plt.savefig('training_curve.png')
print("Graph saved: training_curve.png")