# 2023.0420.0253 @Brian

class StrokeRecognitionDataset:

    def __init__(self, features=None, targets=None):

        self.features = features
        self.targets = targets
        self.classes = {"其他": 0, "右正手發球": 1, "右反手發球": 2, "右正手回球": 3, "右反手回球": 4}
        
    def __len__(self):

        return self.features.shape[0]
    
    def __getitem__(self, idx):

        return self.features[idx], self.targets[idx]