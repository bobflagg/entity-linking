from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn import metrics
import time

class ClusterModel(object):
  """Generic Cluster model."""

  def fit(self, data):
    raise Exception("Method not implemented!")

  def score(self, data):
    return {'description':'Silhouette Coefficient', 'value': metrics.silhouette_score(data, self.labels, metric='euclidean')}

  def visualize(self):
    raise Exception("Method not implemented!")

  def labels(self, threshold=None):
    raise Exception("Method not implemented!")

class KMeansClusterModel(ClusterModel):
  """K-Means Cluster model."""

  def __init__(self, n_clusters, init='k-means++', max_iter=100, n_init=1, verbose=True):
    self.n_clusters = n_clusters
    self.model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init, verbose=verbose)

  def fit(self, data):
    self.model.fit(data)

  def labels(self, threshold=None):
    return model.labels_

  def score(self, data):
    return {'description':'Silhouette Coefficient', 'value': metrics.silhouette_score(data, self.model.labels_, metric='euclidean')}

class HierarchicalClusterModel(ClusterModel):
  """Hierarchical Cluster model."""

  def __init__(self, threshold,  method='complete', criterion='inconsistent', depth=2, R=None, monocrit='MR'):
    self.threshold = threshold
    self.method = method
    self.criterion = criterion
    self.depth = depth
    self.R  = R
    self.monocrit = monocrit

  def fit(self, data):
    print "Fitting HierarchicalClusterModel to data..."
    start = time.time()
    self.Z = linkage(data, method=self.method)
    finish = time.time()
    print '\ttook %0.3f s' % (finish-start)

  def labels(self, threshold=None, criterion=None, depth=None):
    if not threshold: threshold= self.threshold
    if not criterion: criterion= self.criterion
    if not depth: depth= self.depth
    return fcluster(self.Z, threshold, criterion=criterion, depth=depth, R=self.R, monocrit=self.monocrit)

  def visualize(self, threshold=None):
    if not threshold: threshold = self.threshold
    dendrogram(self.Z, color_threshold=threshold, leaf_font_size=6)

# http://nbviewer.ipython.org/github/herrfz/dataanalysis/blob/master/week4/clustering_example.ipynb
# http://stackoverflow.com/questions/16883412/how-do-i-get-the-subtrees-of-dendrogram-made-by-scipy-cluster-hierarchy
# http://stackoverflow.com/questions/11917779/how-to-plot-and-annotate-hierarchical-clustering-dendrograms-in-scipy-matplotlib
# http://stackoverflow.com/questions/15951711/how-to-compute-cluster-assignments-from-linkage-distance-matrices-in-scipy-in-py
