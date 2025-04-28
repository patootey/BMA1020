# choose your own values:
width = 600
height = 480

# choose your own values:
m = 11
n = 9

# generates the grid by 
x_grid, y_grid = np.meshgrid(np.linspace(0,width,m),np.linspace(0,height,n))

# Remember rng, the random number generator from the first question?
# Use rng to generate an (m x n) array of 2d unit vectors, i.e. a (2 x m x n) array
# such that np.sum(vectors ** 2, axis=0) is an array of ones

vectors = # fill in


def noise(x,y,x_grid,y_grid,vectors)
    # fill in code
    return