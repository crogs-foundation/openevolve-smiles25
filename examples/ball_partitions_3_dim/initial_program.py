"""Constructor-based the partition of n-dimensional sphere into (n + 1) parts of smaller diameter"""
# EVOLVE-BLOCK-START

import numpy as np
from scipy.spatial import ConvexHull, distance
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.stats as sps


def construct_packing(n_dim, k_points, radius=0.5):
    """
    Construct a random partition

    Returns:
        points
    """
    # Initialize points via random sampling from the uniform sphere distribution

    points = sps.uniform(loc=-1, scale=2).rvs(
        (k_points, n_dim)
    )  # Используем равномерное распределение
    points = points / np.linalg.norm(points, axis=1, keepdims=True)  # Нормализуем
    points *= radius

    return points


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing(n_dim=3, k_points=4, radius=0.5):
    """Run the n-dimensional sphere partition via points"""
    points = construct_packing(n_dim, k_points, radius)
    return points


def max_cone_diameter(points, cones, radius):
    """Compute the maximal cone's diameter"""
    max_diameter = 0  # Начинаем с диаметра шара

    diams = []
    for cone in cones:
        cone_points = points[cone]
        # distances between cones vertices
        pairwise_dist = [
            distance.euclidean(p1, p2) for p1, p2 in combinations(cone_points, 2)
        ]

        # compute cones diameter
        current_max = max(max(pairwise_dist), radius)
        diams.append(current_max)
        if current_max > max_diameter:
            max_diameter = current_max

    return max_diameter, diams


def visualize(points, cones, radius):
    """
    Visualize the sphere partiion (for two and three-dimensional case)

    Args:
        points, cones, radius
    """
    if points.shape[1] == 2:
        plot_2d_partition(points, cones, radius)
    elif points.shape[1] == 3:
        plot_3d_partition(points, cones, radius)
    else:
        print("Визуализация возможна только в размерностях n_dim = 2, 3")


def plot_2d_partition(points, cones, radius):
    """Визуализация для 2D случая."""
    plt.figure(figsize=(8, 8))
    plt.gca().add_patch(
        plt.Circle((0, 0), radius, fill=False, color="gray", linestyle="--")
    )

    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], "b-")

    for cone in cones:
        for point_idx in cone:
            plt.plot(
                [0, points[point_idx, 0]], [0, points[point_idx, 1]], "r-", alpha=0.3
            )

    plt.scatter(points[:, 0], points[:, 1], c="k", s=50)
    plt.scatter(0, 0, c="g", s=100, label="Центр")
    plt.title("2D разбиение шара на конусы")
    plt.axis("equal")
    plt.legend()
    plt.show()


def plot_3d_partition(points, cones, radius):
    """Визуализация для 3D случая."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Сфера
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x, y, z = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    ax.plot_wireframe(x * radius, y * radius, z * radius, color="gray", alpha=0.2)

    # Выпуклая оболочка
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        for i in range(3):
            start, end = simplex[i], simplex[(i + 1) % 3]
            ax.plot(*points[[start, end]].T, "b-")

    # Конусы
    for cone in cones:
        for point_idx in cone:
            ax.plot(*np.vstack([[0, 0, 0], points[point_idx]]).T, "r-", alpha=0.3)

    ax.scatter(*points.T, c="k", s=50)
    ax.scatter(0, 0, 0, c="g", s=100, label="Центр")
    ax.set_title("3D разбиение шара на конусы")
    ax.legend()
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    n_dim = 3
    k_points = 4
    radius = 0.5

    points = run_packing(n_dim, k_points, radius)
    cones = ConvexHull(points).simplices

    # if n_dim == 2 or n_dim == 3:
    #     visualize(points, cones, radius)

    theoretical_diameter = radius * np.sqrt(2 * (n_dim + 1) / n_dim)
    estimated_diam = max_cone_diameter(points, cones, radius)
    print("Theoretical result:", theoretical_diameter)
    print("Estimated diameter:", estimated_diam)
