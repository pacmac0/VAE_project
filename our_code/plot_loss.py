import json
import matplotlib.pyplot as plt


def read_data_from_file(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def plot_loss(path):
    json = read_data_from_file(path)
    plt.title("Loss per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(json["trainloss"], label="Train-Loss")
    plt.plot(json["trainre"], label="Train-RE")
    plt.plot(json["trainkl"], label="Train-KL")
    plt.plot(json["testloss"], label="Test-Loss")
    plt.plot(json["testre"], label="Test-RE")
    plt.plot(json["testkl"], label="Test-KL")
    plt.legend(("Train-Loss", "Train-RE", "Train-KL", "Test-Loss", "Test-RE", "Test-KL"))
    plt.savefig("plots/plot_{}_{}_loss.png".format(json["config"]["dataset_name"], json["config"]["prior"]))
    plt.show()

if __name__ == "__main__":
    plot_loss(path="experiments/freyfaces/vamp/log.json")