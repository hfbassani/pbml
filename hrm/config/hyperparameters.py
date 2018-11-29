# Hyperparameters
class Hyperparameters():
	def __init__(self,dataset_name='cifar10'):
		super(Hyperparameters, self).__init__()

		self.dataset_name = dataset_name
		self.dataset_path = 'datasets/' + dataset_name
		self.model_path = 'trained_models/'
		self.num_classes = 10
		self.num_epochs =  20
		self.batch_size = 128
		self.learning_rate = 0.01

		## Fashion Mnist
		self.num_epochs = 3

		## CIFAR10
		self.num_epochs =  20
		self.learning_rate = 0.001
		'''
		## CIFAR100
		self.num_classes = 100
		self.learning_rate = 0.001
		self.num_epochs =  50
		'''