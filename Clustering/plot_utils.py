import matplotlib.pyplot as plt

def plot_elbow(wcss, max_k=10):
    plt.plot(range(1, max_k + 1), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()
