import numpy as np
import matplotlib.pyplot as plt
from gen_random_distribution_area import gen_random_distribution_area
from graph import graph
from connectivity_graph import connectivity_graph
from cov_func import cov_func

def main():
    # Network parameters
    covered_area = np.zeros((100, 100))
    obstacle_area = gen_random_distribution_area()
    # max_iterations = 50
    max_iterations = 20
    n_nodes = 40
    # population_size = 50
    population_size = 30
    comm_radius = 10
    sensing_radius = 10
    theta0 = np.pi / 3
    sink = np.array([50, 50])
    scout_bees = population_size
    onlooker_bees = population_size
    acceleration = 1
    l_counts = np.zeros(population_size)
    best_solution = {'position': None, 'cost': 0}
    population = []

    # Initialize population
    for _ in range(population_size):
        alpop = np.column_stack([
            np.random.uniform(sink[0] - comm_radius/2, sink[0] + comm_radius/2, n_nodes),
            np.random.uniform(sink[1] - comm_radius/2, sink[1] + comm_radius/2, n_nodes),
            np.random.uniform(0, 2 * np.pi, n_nodes)
        ])
        alpop[0, :2] = sink
        cost, _ = cov_func(alpop, sensing_radius, theta0, obstacle_area, covered_area.copy())
        population.append({'position': alpop, 'cost': cost})
        if cost > best_solution['cost']:
            best_solution = {'position': alpop.copy(), 'cost': cost}

    # Optimization loop
    for it in range(max_iterations):
        # Scout bee phase
        for i in range(scout_bees):
            k = np.random.randint(population_size)
            phi = acceleration * np.random.uniform(-1, 1, (n_nodes, 3))
            alpop = population[i]['position'] + phi * (population[i]['position'] - population[k]['position'])
            alpop[:, :2] = np.clip(alpop[:, :2], 1, 99)
            alpop[:, 2] = np.mod(alpop[:, 2], 2 * np.pi)
            if connectivity_graph(graph(alpop[:, :2], comm_radius), []):
                alpop_cost, _ = cov_func(alpop, sensing_radius, theta0, obstacle_area, covered_area.copy())
                if alpop_cost >= population[i]['cost']:
                    population[i] = {'position': alpop, 'cost': alpop_cost}
                    if alpop_cost > best_solution['cost']:
                        best_solution = {'position': alpop.copy(), 'cost': alpop_cost}
                else:
                    l_counts[i] += 1

        # Onlooker bee phase
        e_ave = np.array([ind['cost'] for ind in population]) / sum(ind['cost'] for ind in population)
        for _ in range(onlooker_bees):
            i = np.random.choice(population_size, p=e_ave)
            for k in range(n_nodes):
                h = np.random.randint(n_nodes)
                phi = acceleration * np.random.uniform(-1, 1, 3)
                alpop = population[i]['position'].copy()
                alpop[k] += phi * (population[i]['position'][k] - population[i]['position'][h])
                alpop[:, :2] = np.clip(alpop[:, :2], 1, 99)
                alpop[:, 2] = np.mod(alpop[:, 2], 2 * np.pi)
                if connectivity_graph(graph(alpop[:, :2], comm_radius), []):
                    alpop_cost, _ = cov_func(alpop, sensing_radius, theta0, obstacle_area, covered_area.copy())
                    if alpop_cost >= population[i]['cost']:
                        population[i] = {'position': alpop, 'cost': alpop_cost}
                        if alpop_cost > best_solution['cost']:
                            best_solution = {'position': alpop.copy(), 'cost': alpop_cost}
                    else:
                        l_counts[i] += 1

        print(f"Iteration {it + 1}: Best Cost = {best_solution['cost']:.4f}")

    # Visualization
    visualize_solution(best_solution, obstacle_area, covered_area, sensing_radius, theta0)

def visualize_solution(solution, obstacle_area, covered_area, sensing_radius, theta0):
    plt.figure(figsize=(10, 10))
    alpop = solution['position']
    coverage, covered_area = cov_func(alpop, sensing_radius, theta0, obstacle_area, covered_area.copy())
    
    # Plot sensor positions
    for i in range(len(alpop)):
        plt.plot(alpop[i, 1], alpop[i, 0], 'ro', markersize=3, color='red')  # Match MATLAB red color
        plt.text(alpop[i, 1], alpop[i, 0], str(i + 1), fontsize=10, color='red')

    # Plot obstacle area
    obs_row, obs_col = np.where(obstacle_area == 1)
    plt.plot(obs_col, obs_row, '.', markersize=1, color='blue')  # Match MATLAB blue color

    # Plot partially covered area
    obs_row, obs_col = np.where(covered_area == 0.5)
    plt.plot(obs_col, obs_row, '.', markersize=1, color='green')  # Match MATLAB green color

    # Plot fully covered area
    obs_row, obs_col = np.where(covered_area == 1)
    plt.plot(obs_col, obs_row, '.', markersize=2, color='red')  # Match MATLAB red color

    # Adjust axis limits and grid
    plt.grid(True)
    plt.axis('equal')
    plt.xlim([0, obstacle_area.shape[1] - 1])  # Match MATLAB x-axis limits
    plt.ylim([0, obstacle_area.shape[0] - 1])  # Match MATLAB y-axis limits
    plt.title(f'Weighted Coverage ratio: {coverage * 100:.2f}%')
    plt.show()

if __name__ == "__main__":
    main()
