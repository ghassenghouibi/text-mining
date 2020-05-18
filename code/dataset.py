import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def get_len_dataset(self):
    	return len(self.train.data)+len(self.test.data)

    def get_len_train(self):
    	return (len(self.train.data)/self.get_len_dataset())*100.0

    def get_len_test(self):
    	return (len(self.test.data)/self.get_len_dataset())*100.0

    def show_data_len_info(self):
    	print("\nAll   : 100% "+ str(self.get_len_dataset())+"\nTest  :"+ str(self.get_len_test())+"\nTrain :"+ str(self.get_len_train()))

    def viz_partition(self):
    	names=['Train','Test']
    	values=[len(self.train.data),len(self.test.data)]
    	plt.figure(figsize=(4,3))
    	plt.bar(names, values,color=['red', 'green'])
    	plt.xticks(names, ["Train "+str(round(self.get_len_train(),2))+"%","Test "+str(round(self.get_len_test(),2))+"%"])
    	plt.suptitle('Dataset repartition')
    	plt.show()

    def get_data_train(self):
    	return self.train.data

    def data_shape(self):
    	print(self.train.filesnames.shape)

    def get_categories(self):
    	print(self.train.target_names)

    def test_lines(self):
    	numberOfoutlier=0
    	for x in self.train.data:
    		if(x.find('Lines'))==-1:
    			numberOfoutlier+=1
    		
    	return numberOfoutlier

    def test_subject(self):
    	numberOfoutlier=0
    	for x in self.train.data:
    		if(x.find('Subject'))==-1:
    			numberOfoutlier+=1
    	
    	return numberOfoutlier

    def test_organizition(self):
    	numberOfoutlier=0
    	for x in self.train.data:
    		if(x.find('Organization'))==-1:
    			numberOfoutlier+=1
    	
    	return numberOfoutlier

    def test_from(self):
    	numberOfoutlier=0
    	for x in self.train.data:
    		if(x.find('From'))==-1:
    			numberOfoutlier+=1
    	
    	return numberOfoutlier

    def outlier_detect(self):
    	total=self.test_lines()+self.test_subject()+self.test_organizition()+self.test_from()
    	print("Total Outlier found :",total,'\nLines not found in :',self.test_lines(),'\nSubject not found in :',self.test_subject(),'\nOrganization not found in :',self.test_organizition(),'\nFrom not found in :',self.test_from())