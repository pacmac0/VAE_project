#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def plot(logfile, title, figname, fields=['train_loss', 'test_loss']):
    with open(logfile, 'r') as f:
        data = json.load(f)

    losses = {
        'train_loss': data['trainloss'],
        'train_re': data['trainre'],
        'train_kl': data['trainkl'],
        'test_loss': data['testloss'],
        'test_re': data['testre'],
        'test_kl': data['testkl'],
    }

    df = pd.DataFrame(losses)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()
    sns.lineplot(data=df[fields], ax=ax)
    ax.set_title(title, fontsize=16)
    fig.savefig(f'./plots/{figname}.pdf')
