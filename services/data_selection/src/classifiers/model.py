class Model:
    def __init__(self, model, datatype):
        self.dtype = datatype
        self.model = model
    
    def init_model(self, train_data, filter_feature, train_labels=None, label_feature=None):

        
        
        if train_labels is not None:
            filter_vals = train_data[filter_feature].unique()
            input_dims = []
            for val in filter_vals:
                _, dim = filter_vals[filter_feature==val].shape
                input_dims.append(dim)

            self.input_dim=max(input_dims)
        else:


