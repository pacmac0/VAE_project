import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pseudos_plot import plot_pseudos

def save_model(model, name):
    with open(name, "wb") as f:
        torch.save(model, f)

def generate(model, config, epoch):
    samples = model.generate_samples()
    samples = samples.data.cpu().numpy()

    plt.clf()
    plt.close('all')
    plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        sample = sample.reshape(config["input_size"])
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        sample = sample[:, :, 0]
        if (config['input_type'] == 'binary'):
            sample = (sample > 0.5).astype(np.int_)
        plt.imshow(sample, cmap="gray")

    if config["prior"] == "vamp":
        if config["pseudo_from_data"] == True:
            filename = f'experiments/{config["dataset_name"]}/{config["prior"]}/pseudo_from_data' #/epoch_{epoch}'
        else:
            filename = f'experiments/{config["dataset_name"]}/{config["prior"]}/not_pseudo_from_data' #/epoch_{epoch}'
    else:
        filename = f'experiments/{config["dataset_name"]}/{config["prior"]}' #/epoch_{epoch}'
    save_model(model, filename + f"/models/epoch{epoch}.model")
    plt.savefig(filename + f"/images/epoch{epoch}.png")

    if config["prior"] == "vamp":
        plot_pseudos(model, config["input_size"], filename + f"/images/epoch{epoch}_pseudos.png")
