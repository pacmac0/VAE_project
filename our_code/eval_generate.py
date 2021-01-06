import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def generate(modelpath, shape, img_filename):
    if torch.cuda.is_available():
        device = torch.device("gpu")
        with open(modelpath, "rb") as f:
            model = torch.load(f).to(device)
    else:
        device = torch.device("cpu")
        with open(modelpath, "rb") as f:
            model = torch.load(f, map_location=torch.device("cpu")).to(device)

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
        sample = sample.reshape(shape)
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        sample = sample[:, :, 0]
        plt.imshow(sample, cmap="gray")

    plt.savefig(img_filename)
