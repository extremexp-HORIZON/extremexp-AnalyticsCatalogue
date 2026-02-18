import sklearn
from sklearn.tree import DecisionTreeClassifier

class lstmTrees():
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.getDimensions()
        self.clipDataset()

    def getDimensions(self):
        min_time = -1
        for datapoint in self.dataset:
            num_time, num_channels = datapoint.shape
            if min_time == -1:
                min_time = num_time
            
            if num_time < min_time:
                min_time = num_time
        
        self.min_time = min_time

    
    def clipDataset(self):
        new_dataset = []
        for datapoint in self.dataset:
            new_datset = datapoint[-self.min_time:,:]

        self.clipped_dataset = new_dataset


    def getTimestampDataset(self, timestamp):
        timestamp_dataset = []
        for datapoint in self.dataset:
            time_stamp_dp = datapoint[timestamp, :]
            timestamp_dataset.append(time_stamp_dp)
        return timestamp_dataset


        
    def fit(self):
        time_stamp_trees = []
        for i in range(self.min_trees):
            time_stamp_trees.append(DecisionTreeClassifier(class_weight="balanced"))
        

        for timestamp in range(self.min_trees):
            timestamp_data = self.getTimestampDataset(timestamp)
            time_stamp_trees[timestamp].fit(timestamp_data, self.labels)
        
        self.models = time_stamp_trees

    def set_weights(self, weights=None):
        if weights is None:
            res_weights = []
            for i in range(self.min_time + 1, 1, -1):
                res_weights.append(1/(i))
        self.model_weights = res_weights


    def inference(self, datapoint):
        t, n = datapoint.shape
        weighted_res = 0
        for i in range(1, t+1):
            timestamp_pred = self.time_stamp_trees[t -i].predict(datapoint[t-i, :])

            weighted_res += timestamp_pred*self.model_weights[i-1]
        
        return int(weighted_res)