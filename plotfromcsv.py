import matplotlib.pyplot as plt
import csv


num_epochs = 40
def plot_epochs(accuracies, names):
    COLORS = ['xkcd:blurple', 'xkcd:lavender', 'xkcd:lightblue', 'xkcd:indigo', 'xkcd:pink']
    plt.figure()
    epochslist = [i+1 for i in range(num_epochs)]
    for i in range(len(accuracies)):
        plt.plot(epochslist, accuracies[i], COLORS[i], label=names[i])

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0, 100)
    plt.grid()
    plt.legend()

    plt.savefig("accuracy_epoch_plot.png")


f = open("accuracies.csv", "r")
reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)

accuracies = [0, 0, 0, 0, 0]
for index, row in enumerate(reader):
    accuracies[index] = row

f.close()
names = ["basic NN", "GRU (1 layer)", "GRU (3 layer)",  "GRU (repeated layer)", "DEQ"]

plot_epochs(accuracies, names)