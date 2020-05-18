from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import matthews_corrcoef

class TokenizeGroup:
    def __init__(self, data):
    	self.data=data

    def count_vectorizer(self):
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(self.data)
        return X_train_counts

    def tf_idf_transform(self):
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(self.data)
        tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)
        return X_train_tf

    def correlation(self,v1,v2):
        print(matthews_corrcoef(v1,v2))