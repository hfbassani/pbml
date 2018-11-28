# Hyperparameters
class Hyperparameters():
    def __init__(self,dataset_name='cifar10'):
        super(Hyperparameters, self).__init__()

        self.dataset_name = dataset_name
        self.dataset_path = 'datasets/' + dataset_name
        self.model_path = 'trained_models/'
        self.num_classes = 10
        self.num_epochs =  1
        self.batch_size = 100
        self.learning_rate = 0.001

        ## Fashion Mnist
        self.num_epochs = 3

        ## CIFAR10
        self.num_epochs =  10
