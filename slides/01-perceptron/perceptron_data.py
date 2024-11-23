import json

import numpy as np
from numpy.typing import NDArray


def random_nice_dataset(n: int) -> list[tuple[NDArray[np.float64], int]]:
    """Generate a random linearly separable dataset of 2d points with a boolean class.

    ## Arguments

    - `n`: the number of points to generate. Must be at least 2.

    ## Returns

    A `n` length list of tuples `(point, class)`

    - `point` is a numpy NDArray of dtype `np.float64`
    - `class` is an integer, either 0 or 1

    There will be at least one point of each class, and the classes will be separated by a random
    straight line, with a margin of at least 0.5.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, received {n}.")
    rng = np.random.default_rng()
    n_true = rng.integers(low=1, high=n)  # How many items in True
    classes = np.less(np.arange(n), n_true).astype(np.int64)[:, np.newaxis]
    # The idea here is to
    # 1. Generate n points that are all above the x axis as a (n, 2) matrix coord.
    # 2. Select those that will be affected the False class and reflect them round the x axis so
    #    they are under it and we get a problem that is linearly separated by the x axis.
    # 3. Chose a straigtht line D that will be our linear separator. Call theta its angle with the x
    #    axis and beta its ordinate at origin.
    # 4. Transform the data point by rotating around the origin with angle theta and then shifting
    #    them on the y axis from beta. This will make the classification problem linearly separated
    #    by D.
    #    - To do that transformation, we multiply coord to the *right* by the transpose of the theta
    #      rotation matrix.
    #    - Transpose because we will iterate on it and by default iteration on arrays is rowise.
    separator_theta = rng.uniform(-np.pi, np.pi)  # The angle of the linear separtor
    separator_beta = rng.uniform(-8, 8)  # The orginate at origin of the linear separator
    rot_mat_t = np.array(
        [
            [np.cos(separator_theta), np.sin(separator_theta)],
            [-np.sin(separator_theta), np.cos(separator_theta)],
        ],
        dtype=np.float64,
    )
    coord = np.matmul(
        # reflect False items wrt the origin (simpler than just the x axis but doesn't change the
        # distribution)
        (2.0 * classes.astype(np.float64) - 1)
        * (rng.uniform(low=[-8.0, 0.0], high=[8.0, 8.0], size=[n, 2]) + np.array([0.0, 0.5])),
        rot_mat_t,
    ) + np.array([0.0, separator_beta])
    data = np.concatenate(
        (coord, classes),
        axis=1,
    )
    return [(x[:2], x[2].item()) for x in rng.permutation(data)]


iris = (
    [
        [[6.0, 2.2, 4.0, 1.0], "iris-versicolor"],
        [[6.9, 3.1, 5.4, 2.1], "iris-virginica"],
        [[5.5, 2.4, 3.7, 1.0], "iris-versicolor"],
        [[6.3, 2.8, 5.1, 1.5], "iris-virginica"],
        [[6.8, 3.0, 5.5, 2.1], "iris-virginica"],
        [[6.3, 2.7, 4.9, 1.8], "iris-virginica"],
        [[6.3, 3.4, 5.6, 2.4], "iris-virginica"],
        [[5.9, 3.0, 4.2, 1.5], "iris-versicolor"],
        [[6.4, 2.9, 4.3, 1.3], "iris-versicolor"],
        [[5.7, 4.4, 1.5, 0.4], "iris-setosa"],
        [[6.4, 3.2, 4.5, 1.5], "iris-versicolor"],
        [[6.9, 3.2, 5.7, 2.3], "iris-virginica"],
        [[6.1, 2.6, 5.6, 1.4], "iris-virginica"],
        [[4.6, 3.4, 1.4, 0.3], "iris-setosa"],
        [[6.5, 3.0, 5.5, 1.8], "iris-virginica"],
        [[6.9, 3.1, 4.9, 1.5], "iris-versicolor"],
        [[6.7, 2.5, 5.8, 1.8], "iris-virginica"],
        [[5.5, 2.3, 4.0, 1.3], "iris-versicolor"],
        [[7.7, 2.8, 6.7, 2.0], "iris-virginica"],
        [[5.7, 2.6, 3.5, 1.0], "iris-versicolor"],
        [[5.8, 2.8, 5.1, 2.4], "iris-virginica"],
        [[6.3, 2.3, 4.4, 1.3], "iris-versicolor"],
        [[7.7, 2.6, 6.9, 2.3], "iris-virginica"],
        [[6.3, 2.5, 5.0, 1.9], "iris-virginica"],
        [[6.4, 2.7, 5.3, 1.9], "iris-virginica"],
        [[5.1, 3.8, 1.9, 0.4], "iris-setosa"],
        [[6.7, 3.1, 4.7, 1.5], "iris-versicolor"],
        [[5.2, 2.7, 3.9, 1.4], "iris-versicolor"],
        [[5.6, 3.0, 4.5, 1.5], "iris-versicolor"],
        [[4.5, 2.3, 1.3, 0.3], "iris-setosa"],
        [[5.3, 3.7, 1.5, 0.2], "iris-setosa"],
        [[5.1, 3.4, 1.5, 0.2], "iris-setosa"],
        [[6.1, 2.9, 4.7, 1.4], "iris-versicolor"],
        [[5.4, 3.9, 1.3, 0.4], "iris-setosa"],
        [[5.4, 3.9, 1.7, 0.4], "iris-setosa"],
        [[4.4, 3.0, 1.3, 0.2], "iris-setosa"],
        [[5.6, 2.7, 4.2, 1.3], "iris-versicolor"],
        [[6.6, 2.9, 4.6, 1.3], "iris-versicolor"],
        [[4.8, 3.1, 1.6, 0.2], "iris-setosa"],
        [[6.1, 2.8, 4.0, 1.3], "iris-versicolor"],
        [[5.9, 3.0, 5.1, 1.8], "iris-virginica"],
        [[5.5, 2.5, 4.0, 1.3], "iris-versicolor"],
        [[6.7, 3.3, 5.7, 2.1], "iris-virginica"],
        [[5.0, 3.4, 1.6, 0.4], "iris-setosa"],
        [[5.1, 2.5, 3.0, 1.1], "iris-versicolor"],
        [[6.5, 3.0, 5.2, 2.0], "iris-virginica"],
        [[5.5, 4.2, 1.4, 0.2], "iris-setosa"],
        [[5.1, 3.8, 1.6, 0.2], "iris-setosa"],
        [[5.7, 3.8, 1.7, 0.3], "iris-setosa"],
        [[5.7, 3.0, 4.2, 1.2], "iris-versicolor"],
        [[6.2, 3.4, 5.4, 2.3], "iris-virginica"],
        [[4.9, 3.1, 1.5, 0.1], "iris-setosa"],
        [[5.4, 3.4, 1.5, 0.4], "iris-setosa"],
        [[5.1, 3.5, 1.4, 0.3], "iris-setosa"],
        [[4.8, 3.0, 1.4, 0.3], "iris-setosa"],
        [[5.8, 2.7, 5.1, 1.9], "iris-virginica"],
        [[6.9, 3.1, 5.1, 2.3], "iris-virginica"],
        [[6.7, 3.3, 5.7, 2.5], "iris-virginica"],
        [[6.2, 2.8, 4.8, 1.8], "iris-virginica"],
        [[5.0, 3.6, 1.4, 0.2], "iris-setosa"],
        [[7.6, 3.0, 6.6, 2.1], "iris-virginica"],
        [[5.2, 3.5, 1.5, 0.2], "iris-setosa"],
        [[6.1, 3.0, 4.6, 1.4], "iris-versicolor"],
        [[6.0, 2.7, 5.1, 1.6], "iris-versicolor"],
        [[4.9, 2.4, 3.3, 1.0], "iris-versicolor"],
        [[4.8, 3.0, 1.4, 0.1], "iris-setosa"],
        [[7.3, 2.9, 6.3, 1.8], "iris-virginica"],
        [[5.7, 2.8, 4.1, 1.3], "iris-versicolor"],
        [[5.1, 3.8, 1.5, 0.3], "iris-setosa"],
        [[6.7, 3.0, 5.2, 2.3], "iris-virginica"],
        [[5.4, 3.4, 1.7, 0.2], "iris-setosa"],
        [[7.2, 3.0, 5.8, 1.6], "iris-virginica"],
        [[6.3, 2.5, 4.9, 1.5], "iris-versicolor"],
        [[7.7, 3.0, 6.1, 2.3], "iris-virginica"],
        [[5.0, 3.2, 1.2, 0.2], "iris-setosa"],
        [[5.6, 2.8, 4.9, 2.0], "iris-virginica"],
        [[5.0, 3.0, 1.6, 0.2], "iris-setosa"],
        [[5.1, 3.7, 1.5, 0.4], "iris-setosa"],
        [[5.1, 3.3, 1.7, 0.5], "iris-setosa"],
        [[4.6, 3.1, 1.5, 0.2], "iris-setosa"],
        [[6.7, 3.0, 5.0, 1.7], "iris-versicolor"],
        [[5.5, 3.5, 1.3, 0.2], "iris-setosa"],
        [[4.9, 3.0, 1.4, 0.2], "iris-setosa"],
        [[5.0, 2.0, 3.5, 1.0], "iris-versicolor"],
        [[4.4, 3.2, 1.3, 0.2], "iris-setosa"],
        [[7.2, 3.2, 6.0, 1.8], "iris-virginica"],
        [[5.8, 2.6, 4.0, 1.2], "iris-versicolor"],
        [[4.9, 3.1, 1.5, 0.1], "iris-setosa"],
        [[6.1, 3.0, 4.9, 1.8], "iris-virginica"],
        [[5.0, 3.5, 1.6, 0.6], "iris-setosa"],
        [[6.6, 3.0, 4.4, 1.4], "iris-versicolor"],
        [[6.3, 2.9, 5.6, 1.8], "iris-virginica"],
        [[5.9, 3.2, 4.8, 1.8], "iris-versicolor"],
        [[4.9, 3.1, 1.5, 0.1], "iris-setosa"],
        [[6.4, 3.1, 5.5, 1.8], "iris-virginica"],
        [[5.0, 3.3, 1.4, 0.2], "iris-setosa"],
        [[7.7, 3.8, 6.7, 2.2], "iris-virginica"],
        [[6.8, 3.2, 5.9, 2.3], "iris-virginica"],
        [[6.3, 3.3, 6.0, 2.5], "iris-virginica"],
        [[5.5, 2.6, 4.4, 1.2], "iris-versicolor"],
        [[4.9, 2.5, 4.5, 1.7], "iris-virginica"],
        [[5.0, 3.5, 1.3, 0.3], "iris-setosa"],
        [[7.9, 3.8, 6.4, 2.0], "iris-virginica"],
        [[6.5, 3.0, 5.8, 2.2], "iris-virginica"],
        [[6.5, 2.8, 4.6, 1.5], "iris-versicolor"],
        [[5.8, 2.7, 4.1, 1.0], "iris-versicolor"],
        [[7.4, 2.8, 6.1, 1.9], "iris-virginica"],
        [[5.8, 2.7, 3.9, 1.2], "iris-versicolor"],
        [[6.0, 3.4, 4.5, 1.6], "iris-versicolor"],
        [[5.6, 2.9, 3.6, 1.3], "iris-versicolor"],
        [[5.1, 3.5, 1.4, 0.2], "iris-setosa"],
        [[5.4, 3.0, 4.5, 1.5], "iris-versicolor"],
        [[5.8, 4.0, 1.2, 0.2], "iris-setosa"],
        [[5.2, 3.4, 1.4, 0.2], "iris-setosa"],
        [[6.2, 2.9, 4.3, 1.3], "iris-versicolor"],
        [[6.0, 3.0, 4.8, 1.8], "iris-virginica"],
        [[4.4, 2.9, 1.4, 0.2], "iris-setosa"],
        [[4.7, 3.2, 1.6, 0.2], "iris-setosa"],
        [[4.7, 3.2, 1.3, 0.2], "iris-setosa"],
        [[6.8, 2.8, 4.8, 1.4], "iris-versicolor"],
        [[5.7, 2.5, 5.0, 2.0], "iris-virginica"],
        [[4.6, 3.2, 1.4, 0.2], "iris-setosa"],
        [[6.5, 3.2, 5.1, 2.0], "iris-virginica"],
        [[6.3, 3.3, 4.7, 1.6], "iris-versicolor"],
        [[4.8, 3.4, 1.6, 0.2], "iris-setosa"],
        [[5.4, 3.7, 1.5, 0.2], "iris-setosa"],
        [[6.4, 2.8, 5.6, 2.1], "iris-virginica"],
        [[4.8, 3.4, 1.9, 0.2], "iris-setosa"],
        [[5.2, 4.1, 1.5, 0.1], "iris-setosa"],
        [[4.3, 3.0, 1.1, 0.1], "iris-setosa"],
        [[5.6, 3.0, 4.1, 1.3], "iris-versicolor"],
        [[7.2, 3.6, 6.1, 2.5], "iris-virginica"],
        [[5.7, 2.9, 4.2, 1.3], "iris-versicolor"],
        [[6.2, 2.2, 4.5, 1.5], "iris-versicolor"],
        [[6.7, 3.1, 4.4, 1.4], "iris-versicolor"],
        [[6.0, 2.9, 4.5, 1.5], "iris-versicolor"],
        [[5.8, 2.7, 5.1, 1.9], "iris-virginica"],
        [[5.6, 2.5, 3.9, 1.1], "iris-versicolor"],
        [[7.1, 3.0, 5.9, 2.1], "iris-virginica"],
        [[7.0, 3.2, 4.7, 1.4], "iris-versicolor"],
        [[5.0, 2.3, 3.3, 1.0], "iris-versicolor"],
        [[5.0, 3.4, 1.5, 0.2], "iris-setosa"],
        [[6.0, 2.2, 5.0, 1.5], "iris-virginica"],
        [[6.4, 3.2, 5.3, 2.3], "iris-virginica"],
        [[5.7, 2.8, 4.5, 1.3], "iris-versicolor"],
        [[5.5, 2.4, 3.8, 1.1], "iris-versicolor"],
        [[6.4, 2.8, 5.6, 2.2], "iris-virginica"],
        [[4.6, 3.6, 1.0, 0.2], "iris-setosa"],
        [[6.1, 2.8, 4.7, 1.2], "iris-versicolor"],
        [[6.7, 3.1, 5.6, 2.4], "iris-virginica"],
    ],
)
