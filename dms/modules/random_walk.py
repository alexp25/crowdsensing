
import numpy as np
from matplotlib import pyplot as plt
from modules import geometry

class RandomWalk:
    origin = [0, 0]  # starting point
    loc = [0, 0]  # current loc
    boundaries = [0, 0, 0, 0]

    def __init__(self, origin, radius):
        self.origin = origin
        self.loc = origin
    
        self.boundaries[0] = geometry.get_point_on_heading(self.loc, radius, 0)[0]
        self.boundaries[1] = geometry.get_point_on_heading(self.loc, radius, 180)[0]
        self.boundaries[2] = geometry.get_point_on_heading(self.loc, radius, 90)[1]
        self.boundaries[3] = geometry.get_point_on_heading(self.loc, radius, 270)[1]

        # self.boundaries[0] = geometry.get_point_on_heading(self.loc, radius, 0)[1]
        # self.boundaries[1] = geometry.get_point_on_heading(self.loc, radius, 180)[1]
        # self.boundaries[2] = geometry.get_point_on_heading(self.loc, radius, 90)[0]
        # self.boundaries[3] = geometry.get_point_on_heading(self.loc, radius, 270)[0]

        # print(self.origin)
        # print(self.boundaries)

    # square bounded
    def bound_walk_step(self, step):

        def choice(vect):
            index = np.random.choice(range(len(vect)), 1)[0]
            return vect[index]

        # All Possible directiom
        directions = ((-1, 1), (0, 1), (1, 1),
                      (-1, 0), (1, 0),
                      (-1, -1), (0, -1), (1, -1))

        # Directions allowed when x-coordinate reaches boundary
        refelectionsx = ((-1, 1), (0, 1),
                         (-1, 0), (-1, -1), (0, -1))

        # Directions allowed when y-coordinate reaches boundary
        refelectionsy = ((-1, 0), (1, 0),
                         (-1, -1), (0, -1), (1, -1))

        new_point = None

        direction = choice(directions)
        reflection1 = choice(refelectionsx)
        reflection2 = choice(refelectionsy)

        if self.loc[0] > self.boundaries[0]:
            self.loc[0] += reflection1[0] * step
        elif self.loc[0] < self.boundaries[1]:
            self.loc[0] -= reflection1[0] * step
        else:
            self.loc[0] += direction[0] * step

        if self.loc[1] > self.boundaries[2]:
            self.loc[1] += reflection2[1] * step
        elif self.loc[1] < self.boundaries[3]:
            self.loc[1] -= reflection2[1] * step
        else:
            self.loc[1] += direction[1] * step

        new_point = self.loc[:]
        return new_point


if __name__ == "__main__":

    origin = [44.4566428, 26.0808892]
    rw = RandomWalk(origin, 5000)

    # quit()
    points = []
    for i in range(500):
        point = rw.bound_walk_step(0.01)
        # print(point)
        points.append(point)
    
    clat, clon = zip(*[c for c in points])
    plt.plot(clon, clat)
    plt.show()