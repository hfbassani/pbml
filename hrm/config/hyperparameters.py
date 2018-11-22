# Hyperparameters
class Hyperparameters():
    def __init__(self,dataset_name='mnist'):
        super(Hyperparameters, self).__init__()
        
        dataset_name = dataset_name
        num_epochs = 1
        num_classes = 10
        batch_size = 100
        learning_rate = 0.001
        
        self.param = {
            "dataset_name": dataset_name,
            "dataset_path": 'datasets/' + dataset_name,
            "model_path": 'trained_models/',
            "num_classes": num_classes,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
    
    def getParam(self):
        return self.param