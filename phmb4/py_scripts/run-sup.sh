python svm.py  -i ../../Datasets/Realdata_3Times3FoldsExp_Train -p ../Parameters/SVM_0 -o svm
python mlp.py  -i ../../Datasets/Realdata_3Times3FoldsExp_TrainS01 -p ../Parameters/MLP_0 -o mlp -s 0.01
python mlp.py  -i ../../Datasets/Realdata_3Times3FoldsExp_TrainS05 -p ../Parameters/MLP_0 -o mlp -s 0.05
python mlp.py  -i ../../Datasets/Realdata_3Times3FoldsExp_TrainS10 -p ../Parameters/MLP_0 -o mlp -s 0.10
python mlp.py  -i ../../Datasets/Realdata_3Times3FoldsExp_TrainS25 -p ../Parameters/MLP_0 -o mlp -s 0.25
python mlp.py  -i ../../Datasets/Realdata_3Times3FoldsExp_TrainS50 -p ../Parameters/MLP_0 -o mlp -s 0.50
python mlp.py  -i ../../Datasets/Realdata_3Times3FoldsExp_TrainS75 -p ../Parameters/MLP_0 -o mlp -s 0.75
