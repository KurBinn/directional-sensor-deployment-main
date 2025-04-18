import numpy as np

def cov_func(pop, rs, theta0, obstacle_area, covered_area):
    """
    Calculate the coverage ratio and update the covered area.
    """
    covered_area[covered_area != 0] = 0  # Reset covered area
    
    for sensor in pop:
        start_point = np.floor(sensor[:2]).astype(int)
        for i in range(-rs - 1, rs + 2):
            for k in range(-rs - 1, rs + 2):
                map_x, map_y = start_point[1] + i, start_point[0] + k
                if 0 <= map_x < obstacle_area.shape[1] and 0 <= map_y < obstacle_area.shape[0]:
                    dist = np.sqrt((map_y - sensor[0])**2 + (map_x - sensor[1])**2)
                    theta = np.arctan2(map_y - sensor[0], map_x - sensor[1])
                    if theta < 0:
                        theta += 2 * np.pi
                    # Ensure theta is within the sensor's sensing range
                    if dist <= rs and sensor[2] - theta0 / 2 <= theta <= sensor[2] + theta0 / 2:
                        covered_area[map_x, map_y] = obstacle_area[map_x, map_y]

    # Handle obstacles
    obs_row, obs_col = np.where(obstacle_area == 0)
    for x, y in zip(obs_row, obs_col):
        if covered_area[x, y] == 1:
            covered_area[x, y] = -2

    count1 = np.sum(covered_area == 1)
    count2 = np.sum(covered_area == -2)
    count3 = np.sum(obstacle_area == 1)
    coverage = (count1 - count2) / count3
    
    covered_area[covered_area == -2] = -1
    return coverage, covered_area
