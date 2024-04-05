def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def cross_entropy(Y_hat, Y_data):
    delta = 1e-7
    to_sum = Y_data * np.log(Y_hat + delta) + (1 - Y_data) * np.log((1 - Y_hat) + delta)
    return -np.log(to_sum)
