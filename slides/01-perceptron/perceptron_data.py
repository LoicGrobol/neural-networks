import numpy as np
from numpy.typing import NDArray


def random_nice_dataset(
    n: int, bias: bool = True, _domain_radius: float = 8.0
) -> list[tuple[NDArray[np.float64], int]]:
    """Generate a random linearly separable dataset of 2d points with a boolean class.

    ## Arguments

    - `n`: the number of points to generate. Must be at least 2.
    - `bias` whether the linear separator has a bias.

    ## Returns

    A `n` length list of tuples `(point, class)`

    - `point` is a numpy NDArray of dtype `np.float64`
    - `class` is an integer, either 0 or 1

    Each class will be at least 1/4 of the points and no less than 1, and the classes will be
    separated by a random straight line.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, received {n}.")
    rng = np.random.default_rng()
    min_class_size = max(1, n//4)
    n_true = rng.integers(low=min_class_size, high=n-min_class_size)  # How many items in True
    classes = rng.permutation(np.less(np.arange(n), n_true).astype(np.int64))
    # The idea here is to
    # 1. Chose a directed straight line D that will be our pre-shift linear separator. Call θ its
    #    angle with the x axis
    # 2. Generate n points in polar coordinates $(ρ, η)$ with $η∈]0, π[$.
    # 3. Reflect the angles of the 0 class so they are in $]-π, 0[$
    # 4. Add θ to all the angles
    # 5. Shift all the points' ordinates by the bias value β
    theta = rng.uniform(-np.pi, np.pi)  # The angle of the linear separtor
    radii = rng.uniform(low=np.nextafter(0.0, 1.0), high=_domain_radius, size=n)
    # Flipping the False items
    angles = (
        rng.uniform(low=np.nextafter(0, 1.0), high=np.pi, size=n)
        * (2.0 * classes.astype(np.float64) - 1)
        + theta
    )
    # This could be more compact but who cares (me, i do care, but it's ok this time)
    if bias:
        beta = rng.uniform(-1.0, 1.0)
    else:
        beta = 0.0
    coord = np.stack(
        (
            radii * np.cos(angles),
            radii * np.sin(angles) + beta,
        ),
        axis=1,
    )
    return [(x, y.item()) for x, y in zip(coord, classes, strict=True)]


def random_okay_dataset(
    n: int, _domain_radius: float = 8.0
) -> list[tuple[NDArray[np.float64], int]]:
    """Generate a random non-linearly separable dataset of 2d points with a boolean class.

    ## Arguments

    - `n`: the number of points to generate. Must be at least 4.
    - `bias` whether the linear separator has a bias.

    ## Returnss

    A `n` length list of tuples `(point, class)`

    - `point` is a numpy NDArray of dtype `np.float64`
    - `class` is an integer, either 0 or 1

    Each class will be at least 1/4 of the points and no less than 1, and the classes will not be
    separable by a straight line.
    """
    if n < 4:
        raise ValueError(f"n must be at least 4, received {n}.")
    rng = np.random.default_rng()
    min_class_size = max(1, n//4)
    n_true = rng.integers(low=min_class_size, high=n-min_class_size)  # How many items in True
    classes = rng.permutation(np.less(np.arange(n), n_true).astype(np.int64))

    theta = rng.uniform(-np.pi, np.pi)  # The angle of the linear separtor
    radii = rng.uniform(low=np.nextafter(0.0, 1.0), high=_domain_radius, size=n)
    # Flipping the False items
    angles = (
        rng.uniform(low=np.nextafter(0, 1.0), high=np.pi, size=n)
        * (2.0 * classes.astype(np.float64) - 1)
        + theta
    )
    # This could be more compact but who cares (me, i do care, but it's ok this time)
    if bias:
        beta = rng.uniform(-1.0, 1.0)
    else:
        beta = 0.0
    coord = np.stack(
        (
            radii * np.cos(angles),
            radii * np.sin(angles) + beta,
        ),
        axis=1,
    )
    return [(x, y.item()) for x, y in zip(coord, classes, strict=True)]


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
