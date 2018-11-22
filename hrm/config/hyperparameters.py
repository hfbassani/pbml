# Hyperparameters
class Hyperparameters():
    def __init__(self):
        super(Hyperparameters, self).__init__()
        
        num_epochs = 5
        num_classes = 10
        batch_size = 100
        learning_rate = 0.001
        
        self.param = {
            "dataset_path": 'data/mnist',
            "num_classes": num_classes,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
    
    def getParam(self):
        return self.param