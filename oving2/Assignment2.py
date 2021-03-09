import numpy as np
import matplotlib.pyplot as plt

pi = np.array([0.5, 0.5])  # initial probability of the states
states = ["fish", "nofish"]
observations = ["birds", "nobirds"]

transition_probability = np.array([(0.8, 0.2), (0.3, 0.7)])  # Probability of transition between the states matrix

oberservation_true = np.array([(0.75, 0.0), (0.0, 0.2)])  # Probability of observation when state is true
observation_false = np.array([(0.25, 0.0), (0.0, 0.8)])  # Probability of observation when state is false

evidence = [1, 1, 0, 1, 0, 1]  # The previous six states
observation = [observation_false, oberservation_true]


# task 1b)

def forward_eq_a():
    results = np.zeros((7, 2))
    f = pi
    results[0] = f
    for i in range(1, 7):
        obs = observation[evidence[i - 1]]
        f_new = obs @ transition_probability.T @ f
        f = f_new
        norm = sum(f)
        res = np.array(f) / norm
        results[i] = res
    x_values = [0, 1, 2, 3, 4, 5, 6]
    plt.plot(x_values, results[:, 0], label="Fish")
    plt.plot(x_values, results[:, 1], label="No Fish")
    plt.legend()
    plt.show()

    return results

# task 1c


def forward_eq_b():
    f = forward_eq_a()[-1]
    print(f)
    results = np.zeros((26, 2))
    for i in range(0, 26):
        f_new = transition_probability.T @ f
        f = f_new
        norm = sum(f)
        res = np.array(f) / norm
        results[i] = res
    x_values = range(4, 30)
    plt.plot(x_values, results[:, 0], label="Fish")
    plt.plot(x_values, results[:, 1], label="No Fish")
    plt.legend()
    plt.show()
    return results

# task 1d


def smoothing():
    forw = forward_eq_a()
    back = np.zeros((7, 2))
    results = np.zeros((6, 2))
    b = np.array([1, 1])
    for i in range(6, -1, -1):
        back[i] = b
        b_new = transition_probability @ observation[evidence[i-1]] @ b
        b = b_new
    for i in range(5, -1, -1):
        p = forw[i] * back[i]
        norm = sum(p)
        res = np.array(p) / norm
        results[i] = res
    x_values = range(0, 6)
    plt.plot(x_values, results[:, 0], label="Fish")
    plt.plot(x_values, results[:, 1], label="No Fish")
    plt.legend()
    plt.show()
    return results

def viterbi_alg():
    path = np.zeros((6, 2))
    init = forward_eq_a()[1]
    path[0] = init
    parents = np.zeros((6, 2))
    for i in range(1, 6):
        if evidence[i]:
            #print(0.75*0.80*path[i-1][0])
            #rint(path[i-1])
            fish = np.array([0.75*0.80*path[i-1][0], 0.75*0.30*path[i-1][1]])
            no_fish = np.array([0.20*0.20*path[i-1][0], 0.20*0.70*path[i-1][1]])
            prob = np.array([np.max(fish), np.max(no_fish)])
            parent = np.array([np.argmax(fish), np.argmax(no_fish)])
            path[i] = prob
            parents[i] = parent
        else:
            fish = np.array([0.25*0.80*path[i-1][0], 0.25*0.30*path[i-1][1]])
            no_fish = np.array([0.80*0.20*path[i-1][0],  0.80*0.70*path[i-1][1]])
            prob = np.array([np.max(fish), np.max(no_fish)])
            parent = np.array([np.argmax(fish), np.argmax(no_fish)])
            path[i] = prob
            parents[i] = parent
    print(path)
    print(parents)











if __name__ == '__main__':
    print(forward_eq_a())
    print(forward_eq_b())
    print(smoothing())
    viterbi_alg()






