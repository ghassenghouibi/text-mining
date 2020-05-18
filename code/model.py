from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

class Models:



	def naive_bayes(self,test,data,target,categories):
		print("-----------------Naive-Bayes------------------------")
		count_vect = CountVectorizer()
		X_train_counts = count_vect.fit_transform(data)
		tfidf_transformer = TfidfTransformer()
		X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

		clf = MultinomialNB().fit(X_train_tfidf, target)
		X_new_counts = count_vect.transform(test)
		X_new_tfidf = tfidf_transformer.transform(X_new_counts)
		predicted = clf.predict(X_new_tfidf)
		print("Predicted categories is :",categories[predicted[0]],"with score %",round(clf.score(X_train_tfidf, target)*100.0,2))

	def knn(self,test,data,target,categories):
		print("-----------------KNN--------------------------------")
		count_vect = CountVectorizer()
		X_train_counts = count_vect.fit_transform(data)
		tfidf_transformer = TfidfTransformer()
		X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

		clf = KNeighborsClassifier(n_neighbors=7).fit(X_train_tfidf, target)
		X_new_counts = count_vect.transform(test)
		X_new_tfidf = tfidf_transformer.transform(X_new_counts)
		predicted = clf.predict(X_new_tfidf)
		print("Predicted categories is :",categories[predicted[0]],"with score  %",round(clf.score(X_train_tfidf, target)*100.0,2))

	def sgdc(self,test,data,target,categories):
		print("-----------------SGDClassifier----------------------")
		count_vect = CountVectorizer()
		X_train_counts = count_vect.fit_transform(data)
		tfidf_transformer = TfidfTransformer()
		X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

		clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=200, tol=None).fit(X_train_tfidf, target)
		X_new_counts = count_vect.transform(test)
		X_new_tfidf = tfidf_transformer.transform(X_new_counts)
		predicted = clf.predict(X_new_tfidf)
		print("Predicted categories is :",categories[predicted[0]],"with score  %",round(clf.score(X_train_tfidf, target)*100.0,2))

	def svm(self,test,data,target,categories):
		print("-----------------SVM--------------------------------")
		count_vect = CountVectorizer()
		X_train_counts = count_vect.fit_transform(data)
		tfidf_transformer = TfidfTransformer()
		X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

		clf = LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.01,
     verbose=0).fit(X_train_tfidf, target)
		X_new_counts = count_vect.transform(test)
		X_new_tfidf = tfidf_transformer.transform(X_new_counts)
		predicted = clf.predict(X_new_tfidf)
		print("Predicted categories is :",categories[predicted[0]],"with score  %",round(clf.score(X_train_tfidf, target)*100.0,2))

	def linear_regression(self,test,data,target,categories):
		#TODO
		count_vect = CountVectorizer()
		X_train_counts = count_vect.fit_transform(data)
		tfidf_transformer = TfidfTransformer()
		X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
		clf = LinearRegression().fit(X_train_tfidf, target)
		X_new_counts = count_vect.transform(test)
		X_new_tfidf = tfidf_transformer.transform(X_new_counts)
		predicted = clf.predict(X_new_tfidf)
		print(int(predicted))
		indice=int(predicted[0])
		print("Predicted categories is :",categories[indice],"with score  %",round(clf.score(X_train_tfidf, target)*100.0,2))

