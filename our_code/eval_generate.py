import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.utils.data as data_utils

with open('./snapshots/model.model', 'rb') as f:
    model = torch.load(f)

samples = model.generate_x()
samples = samples.data.cpu().numpy()

def visualize_generated_samples(samples):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample = sample.reshape(1, 28, 28)
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        sample = sample[:, :, 0]
        plt.imshow(sample, cmap='gray')

    plt.show()


def visualize_latent_space():
    # start processing (load presplitted data sets)
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32')
    y_test = np.zeros( (x_test.shape[0], 1) )

    # Load test dataset
    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=100, shuffle=True)

    for i, data in enumerate(test_loader, 0):
        image, _ = data

        z_mean, z_logvar = model.get_latent(image)
        sample = z_mean[0]
        print(sample)
        break

if __name__ == '__main__':
    # visualize_latent_space()
    visualize_generated_samples(samples)