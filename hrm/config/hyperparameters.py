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
			self.num_epochs = 10
			self.learning_rate = 0.001
			self.batch_size = 64
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
			self.num_epochs =  25

	def printHyper(self):
		print("Dataset Name: " + str(self.dataset_name))
		print("Dataset Path: " + str(self.dataset_path))
		print("Model Path: " + str(self.model_path))
		print("Num Classes: " + str(self.num_classes))
		print("Num Epochs: " + str(self.num_epochs))
		print("Batch Size: " + str(self.batch_size))
		print("Learning Rate: " + str(self.learning_rate))

	def printHyper(self,file):
		file.write("Dataset Name: " + str(self.dataset_name) + '\n')
		file.write("Dataset Path: " + str(self.dataset_path) + '\n')
		file.write("Model Path: " + str(self.model_path) + '\n')
		file.write("Num Classes: " + str(self.num_classes) + '\n')
		file.write("Num Epochs: " + str(self.num_epochs) + '\n')
		file.write("Batch Size: " + str(self.batch_size) + '\n')
		file.write("Learning Rate: " + str(self.learning_rate) + '\n')
