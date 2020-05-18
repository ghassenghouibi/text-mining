from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from dataset import *
from textprocessing import *
from model import *
from visualization import *

twenty_train = fetch_20newsgroups(subset='train', shuffle=False, random_state=42)
twenty_test  = fetch_20newsgroups(subset='test' , shuffle=False, random_state=42)



data=Dataset(twenty_train,twenty_test)



tokenize=TokenizeGroup((data.get_data_train()))

model=Models()

#model.naive_bayes([twenty_test.data[19]],twenty_train.data,twenty_train.target,twenty_train.target_names)
#model.knn([twenty_test.data[19]],twenty_train.data,twenty_train.target,twenty_train.target_names)
#model.sgdc([twenty_test.data[19]],twenty_train.data,twenty_train.target,twenty_train.target_names)
#model.svm([twenty_test.data[19]],twenty_train.data,twenty_train.target,twenty_train.target_names)



#model.naive_bayes(["God is love"],twenty_train.data,twenty_train.target,twenty_train.target_names)
m#odel.knn(["I want to buy a new car"],twenty_train.data,twenty_train.target,twenty_train.target_names)
#model.sgdc(["We always win in games "],twenty_train.data,twenty_train.target,twenty_train.target_names)
#model.svm(["I want to buy a new car "],twenty_train.data,twenty_train.target,twenty_train.target_names)



viz=ShowPlots()
viz.plot_kmeans()
