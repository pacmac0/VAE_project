import torch
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
    fig = plt.figure(figsize=(5, 5))
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
        plt.imshow(sample, cmap="gray")

    filename = f'experiments/{config["dataset_name"]}/{config["prior"]}/epoch_{epoch}'
    save_model(model, filename + ".model")
    plt.savefig(filename + ".png")

    if config["prior"] == "vamp":
        plot_pseudos(model, config["input_size"], filename + "_pseudos.png")
