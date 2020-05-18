import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_20newsgroups

class ShowPlots:

	def plot_kmeans(self):
		"""Some code here adapted from : https://stackoverflow.com/questions/27494202/how-do-i-visualize-data-points-of-tf-idf-vectors-for-kmeans-clustering/41980547 """
		num_clusters = 3
		num_seeds = 10
		max_iterations = 500
		labels_color_map = {0: '#ff0000', 1: '#00ff00', 2: '#0000ff'}
		pca_num_components = 2
		tsne_num_components = 2

		cats=['comp.sys.mac.hardware','rec.autos','soc.religion.christian']
		three_test=fetch_20newsgroups(subset='test',shuffle=False,random_state=42,categories=cats)
		texts_list=three_test.data[:200]

		tf_idf_vectorizer = TfidfVectorizer(analyzer="word", use_idf=True, smooth_idf=True, ngram_range=(2, 3))
		tf_idf_matrix = tf_idf_vectorizer.fit_transform(texts_list)

		clustering_model = KMeans(
		    n_clusters=num_clusters,
		    max_iter=max_iterations,
		    precompute_distances="auto",
		    n_jobs=-1
		)

		labels = clustering_model.fit_predict(tf_idf_matrix)

		X = tf_idf_matrix.todense()

		reduced_data = PCA(n_components=pca_num_components).fit_transform(X)

		fig, ax = plt.subplots()
		for index, instance in enumerate(reduced_data):
		    pca_comp_1, pca_comp_2 = reduced_data[index]
		    color = labels_color_map[labels[index]]
		    ax.scatter(pca_comp_1, pca_comp_2, c=color)
		plt.show()



	