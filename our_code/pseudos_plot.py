import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# only works for vampprior 
def plot_pseudos(model, shape, img_filename):
    # N = 25 number of pseudos (out of K=500) standard setting
    pseudo_inputs = model.get_pseudos()

    pseudo_inputs = pseudo_inputs.data.cpu().numpy()

    plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(pseudo_inputs):
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
    plt.clf()
