from sklearn import datasets

def getMoon(n_sample=100,noise=0):
    return datasets.make_moons(n_samples= n_sample) if noise == 0 else datasets.make_moons(
        n_samples=n_sample, noise=noise)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    x,y = getMoon()
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.show()