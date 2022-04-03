import numpy as np
import argparse
import matplotlib.pyplot as plt


def y_function(k, x, alpha, a=100):
    return a + k * np.exp(alpha * x)


def main(args):
    alpha = args.alpha
    k = args.k

    xmax = args.xmax
    x_vals = []
    y_vals = []
    for i in np.arange(1, xmax, 0.5):
        x_vals.append(i)
        y_vals.append(y_function(k, i, alpha))

    # plt.plot(x_vals,y_vals,marker='o')
    plt.plot(x_vals, y_vals)
    plt.title(r"y=$" + str(k) + "x^{" + str(alpha) + "}$")
    # plt.title("y = kx^(alpha); alpha = {} k = {}".format(alpha, k))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("class_hists/img_k={}_alpha={}.png".format(k, alpha))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0, help="Law's exponent")
    parser.add_argument("--k", "--K", type=float, default=1.0, help="Constant")
    parser.add_argument("--xmax", type=int, default=10, help="Maximum value on x-axis")
    args = parser.parse_args()
    main(args)
