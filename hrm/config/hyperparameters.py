# Hyperparameters
class Hyperparameters():
	def __init__(self,dataset_name='mnist'):
		super(Hyperparameters, self).__init__()

		self.dataset_name = dataset_name
		self.dataset_path = 'datasets/' + dataset_name
		self.model_path = 'trained_models/'
		
		self.num_classes = 10
		self.num_epochs =  20
		self.batch_size = 128
		self.learning_rate = 0.01

		if(dataset_name == 'mnist'):
			self.num_epochs =  3
		elif(dataset_name == 'fashion_mnist'):
			## Fashion Mnist
			self.num_epochs = 3
		elif(dataset_name == 'svhn'):
			## Fashion Mnist
			self.num_epochs = 20
			self.learning_rate = 0.001
		elif(dataset_name == 'cifar10'):
			## CIFAR10
			self.num_epochs =  20
			self.learning_rate = 0.001
		elif(dataset_name == 'cifar100'):
			## CIFAR100
			self.num_classes = 100
			self.learning_rate = 0.001
			self.num_epochs =  50

	def printHyper(self):
		print("" + str(self.dataset_name))
		print("" + str(self.dataset_path))
		print("" + str(self.model_path))
		print("" + str(self.num_classes))
		print("" + str(self.num_epochs))
		print("" + str(self.batch_size))
		print("" + str(self.learning_rate))
