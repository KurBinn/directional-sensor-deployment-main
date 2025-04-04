import numpy as np

def gen_random_distribution_area():
    """
    Generate a random distribution area with obstacles.
    """
    obstacle_area = np.ones((100, 100)) / 2
    locations = [[50, 50]]
    values = [1]
    for loc, val in zip(locations, values):
        obstacle_area[loc[0], loc[1]] = val
        for j in range(31):
            for k in range(31):
                if 20 <= np.sqrt(j**2 + k**2) <= 30:
                    for dx, dy in [(j, k), (j, -k), (-j, k), (-j, -k)]:
                        x, y = loc[0] + dx, loc[1] + dy
                        if 0 <= x < obstacle_area.shape[0] and 0 <= y < obstacle_area.shape[1]:
                            obstacle_area[x, y] = 1
    return obstacle_area
