import urllib.request
import os

# open a connection to a URL using urllib
data_sets = {
    "binarized_mnist_train.amat": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat",
    "binarized_mnist_valid.amat": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat",
    "binarized_mnist_test.amat": "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat",
}

for set_name in data_sets:
    webUrl = urllib.request.urlopen(data_sets[set_name])

    # get the result code and print it
    print(set_name + "result code: " + str(webUrl.getcode()))

    with open(os.path.join("datasets", "MNIST_static", set_name), "wb") as f:
        f.write(webUrl.read())
