import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class Generator():
    def __init__(self,
                 r: Tuple[int, int] = (-1000, 1000),
                 centroids_num: int = 3,
                 points_num: int = 300) -> None:
        self.range = r
        self.cen_num = centroids_num
        self.points_num = points_num

        self.data = []

    def generate(self) -> None:
        self.centers = np.random.uniform(self.range[0],
                                         self.range[1],
                                         size=(self.cen_num, 2))

        for center in self.centers:
            point = center + np.random.randn(self.points_num, 2) * 100
            self.data.append(point)

        self.data = np.vstack(self.data)
        print(self.data)
        np.savetxt("dataset.csv", self.data, delimiter=',')

    def plot(self) -> None:
        plt.scatter(self.centers[:, 0], self.centers[:, 1], marker="x")
        plt.scatter(self.data[:, 0], self.data[:, 1])

        plt.savefig('data_set.jpg')
        plt.show()


class K_means:
    def __init__(self, data: np.ndarray) -> None:
        self.points = data

    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**(1/2)

    def start_k_means(self, k: int) -> None:
        start_time = time.time() 
        g_k = [[] for _ in range(k)]
        x_min = min(p[0] for p in self.points)
        x_max = max(p[0] for p in self.points)
        y_min = min(p[1] for p in self.points)
        y_max = max(p[1] for p in self.points)
        c_k = [[random.uniform(x_min, x_max),
                random.uniform(y_min, y_max)] for _ in range(k)]

        while True:
            g_k = [[] for _ in range(k)]
            for point in self.points:
                distances = [self.distance(point, centroid)
                             for centroid in c_k]
                selected_centroid = distances.index(min(distances))
                g_k[selected_centroid].append(point)

            new_c = []

            converged = True
            for i, g in enumerate(g_k):
                if not g:
                    new_c.append(random.choice(self.points))
                    continue

                mean = [
                    sum(point[0] for point in g) / len(g),
                    sum(point[1] for point in g) / len(g)
                ]
                new_c.append(mean)

                if self.distance(mean, c_k[i]) > 0.05:
                    converged = False

            c_k = new_c

            if converged:
                break
        end = time.time()
        logger.info("Used time is %s:", end - start_time)

        # DEBUG
        logger.info("plotting...")
        colors = "rgby"
        for i, c in enumerate(c_k):
            plt.scatter(c[0], c[1], color="black", marker='x', zorder=3)
        for i, g in enumerate(g_k):
            for point in g:
                plt.scatter(point[0], point[1], color=colors[i], zorder=1)
        plt.savefig("Result1.jpg")
        plt.show()


if __name__ == "__main__":
    # generator = Generator(centroids_num=4)
    # generator.generate()
    # generator.plot()

    data = np.genfromtxt("dataset.csv", delimiter=',')
    k_means = K_means(data=data)
    k_means.start_k_means(4)
