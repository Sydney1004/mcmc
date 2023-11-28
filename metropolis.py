import numpy as np
import matplotlib.pyplot as plt

def likelihood(x, sd = 0.5, mu = 0):
    '''
    target distribution, current one is log normal
    '''
    val1 = (1/(x*sd*np.sqrt(2*np.pi)))
    val2 = (np.exp((-1*((np.log(x) - mu)**2))/(2*(sd**2))))
    if ((1/(x*np.sqrt(2*np.pi)))*(np.exp((-1*((np.log(x) - mu)**2))/(2*(sd**2)))) == 0):
        print(f"val1: {val1}")
        print(f"val2: {val2}")
        print(f"x: {x}")

    return (1/(x*sd*np.sqrt(2*np.pi)))*(np.exp((-1*((np.log(x) - mu)**2))/(2*(sd**2))))

def uniform_threshold():
    return np.random.uniform(0, 1)

def sample_from_normal(mean):
    '''
    using normal distirbution as the proposal
    '''
    while True:
        sample = np.random.normal(mean, 1)
        if(sample > 0):
            return sample
#     return np.random.exponential(mean)
#     return np.random.normal(mean, 1)

def accept_likelihood(current_candidate, next_candidate):
    assert(likelihood(next_candidate) != 0)
    assert(likelihood(current_candidate) != 0)
    return likelihood(next_candidate)/likelihood(current_candidate)

def sample(n, initial):
    '''
    sample n times with a initial value
    '''
    start = initial
    samples = [start]
    for i in range(n):
        current_candidate = samples[-1]
        next_candidate = sample_from_normal(current_candidate)
        threshold = uniform_threshold()
        if (accept_likelihood(current_candidate, next_candidate) > threshold):
            samples.append(next_candidate)
        # if (accept_likelihood(current_candidate, next_candidate) > np.log(threshold)):
        #     samples.append(next_candidate)
        else:
            samples.append(current_candidate)
    # return samples[int(n*0.2):] why this?
    return samples

x = sample(100000, 5)
# x += sample(100000, )
# x += sample(100000, 0.5)
# x += sample(100000, 0.25)
# x += sample(100000, 1)
print(f"mean: {np.mean(x)}, sd: {np.std(x)}")
plt.hist(x)
plt.show()
