from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def PCA_BestFeatures(x_test, y_test, y_bar):
    pca = PCA(2)
    pca.fit(x_test)
    p = pca.transform(x_test)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(p[:, 0], p[:, 1], y_test, '*',label = 'test_data')
    ax.scatter(p[:, 0], p[:, 1], y_bar, '.',label = 'prediction')
    ax.legend(loc = 'upper right')
    plt.show()
