import datetime
import os.path as osp
import torch
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from adam_optimizer_original import AdamNormGrad
import torch.utils.data as data_utils
from VAE import VAE

# config
config = {
    'seed': 14,
    'dataset_name': 'static_mnist',
    'model_name': 'vae',
    'prior': 'standard',
    'number_components': 500,
    'warmup': 100,
    'z1_size': 40,
    'z2_size': 40,
    'batch_size': 100,
    'test_batch_size': 100,
    'input_size': [1, 28, 28],
    'input_type': 'binary',
    'dynamic_binarization': False,
    'use_training_data_init': 1,
    'pseudoinputs_std': 0.01,
    'pseudoinputs_mean': 0.05,
    'learning_rate': 0.0005,
    'max_epoch': 2000,
}

# loading data (static mnist)
def load_static_mnist(args):
    # set args
    args['input_size'] = [1, 28, 28]
    args['input_type'] = 'binary'
    args['dynamic_binarization'] = False

    # start processing (load presplitted data sets)
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('datasets', 'MNIST_static', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32')

    # shuffle train data
    np.random.shuffle(x_train)

    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args['batch_size'], shuffle=True)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args['test_batch_size'], shuffle=False)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args['test_batch_size'], shuffle=True)

    # setting pseudo-inputs inits
    if args['use_training_data_init'] == 1:
        args['pseudoinputs_std'] = 0.01
        init = x_train[0:args['number_components']].T
        args['pseudoinputs_mean'] = torch.from_numpy( init + args['pseudoinputs_std'] * np.random.randn(np.prod(args['input_size']), args['number_components']) ).float()
    else:
        args['pseudoinputs_mean'] = 0.05
        args['pseudoinputs_std'] = 0.01

    return train_loader, val_loader, test_loader, args


# usage from main: plot_tensor(inputs[0])
def plot_tensor(tensor):
    nparr = tensor.numpy()
    img = np.reshape(nparr, (28, 28))
    plt.figure()
    plt.imshow(img)
    plt.show()


def main(args):
    torch.manual_seed(14)
    model_name = args['dataset_name'] + '_' + args['model_name'] + '_' + args['prior'] + '(K_' + str(args['number_components']) + ')' + '_wu(' + str(args['warmup']) + ')' + '_z1_' + str(args['z1_size']) + '_z2_' + str(args['z2_size'])
    print(args)
    train_loader, val_loader, test_loader, args = load_static_mnist(args)

    # Check if a trained model exists
    if osp.exists('./snapshots/model.model'):
        with open('./snapshots/model.model', 'rb') as f:
            model = torch.load(f)
        print('--> Loaded from pretrained model')
    else:  # Otherwise create and intialize a new model
        model = VAE(args)
        print('--> Initialized new model')

    model.train()
    optimizer = AdamNormGrad(model.parameters(), lr=args['learning_rate'])

    for epoch in range(1, args['max_epoch']):
        # Warm up
        beta = 1.* epoch / args['warmup']
        if beta > 1.:
            beta = 1.
        print(f'--> beta: {beta}')

        train_loss = []
        train_re = []
        train_kl = []

        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            # print("\nTraining batch #", i)
            # get input, data as the list of [inputs, label]
            inputs, _ = data
            mean_dec, logvar_dec, z, mean_enc, logvar_enc = model.forward(inputs)
            loss, RE, KL = model.get_loss(inputs, mean_dec, z, mean_enc, logvar_enc, beta=beta)
            loss.backward()

            if i == len(train_loader) / 2:
                print('loss', loss.item(), 'RE', RE.item(), 'KL', KL.item())
            optimizer.step()
            train_loss.append(loss.item())
            train_re.append(RE.item())
            train_kl.append(KL.item())
        
        epoch_loss = sum(train_loss) / len(train_loader)
        epoch_re = sum(train_re) / len(train_loader)
        epoch_kl = sum(train_kl) / len(train_loader)

        print(f'Epoch: {epoch}; loss: {epoch_loss}, RE: {epoch_re}, KL: {epoch_kl}')

        with open('./snapshots/model.model', 'wb') as f:
            torch.save(model, f)

if __name__ == '__main__':
    torch.set_num_threads(8)
    main(config)

