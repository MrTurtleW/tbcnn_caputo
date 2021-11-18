import matplotlib.pyplot as plt

frac_accuracy = 'run_caputo-tag-accuracy_1.csv'
frac_loss = 'run_caputo-tag-loss_1.csv'

int_accuracy = 'run_integer-tag-accuracy_1.csv'
int_loss = 'run_integer-tag-loss_1.csv'

smooth_count = 5


def smooth(y, index):
    result = []
    for i in range(-smooth_count, smooth_count):
        current_index = index + i
        if 0 <= current_index < len(y):
            try:
                result.append(float(y[current_index]))
            except ValueError:
                pass

    return sum(result) / len(result)


def get_data(filename):
    x = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            time, step, value = line.split(',')
            try:
                float(value)
            except ValueError:
                continue
            x.append(step)
            y.append(value)

    for index in range(len(y)):
        y[index] = smooth(y, index)
    return x, y


def plot_accuracy():
    frac_acc_x, frac_acc_y = get_data(frac_accuracy)
    int_acc_x, int_acc_y = get_data(int_accuracy)
    plt.plot(frac_acc_x, frac_acc_y, label='tbcnn-caputo')
    plt.plot(int_acc_x, int_acc_y, label='tbcnn')
    x_major_locator = plt.MultipleLocator(100)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(loc='best')
    plt.xlabel('step')
    plt.ylabel('accuracy')
    plt.show()


def plot_loss():
    frac_loss_x, frac_loss_y = get_data(frac_loss)
    int_loss_x, int_loss_y = get_data(int_loss)

    plt.plot( frac_loss_x, frac_loss_y, label='tbcnn-caputo')
    plt.plot(int_loss_x, int_loss_y, label='tbcnn')

    plt.legend(loc='best')
    x_major_locator = plt.MultipleLocator(100)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    # plot_loss()
    plot_accuracy()