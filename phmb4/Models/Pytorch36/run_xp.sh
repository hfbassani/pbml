tmux new-session -d "conda activate deepunsupervised & python train_sssom.py --batch-size 32 --start-idx 0 --stop-idx 10 -i ../../Parameters/batch_sssom_uci/diabetes/inputPathsReal_train_diabetes -t ../../Parameters/batch_sssom_uci/diabetes/inputPathsReal_test_diabetes -r ../../Parameters/batch_sssom_uci/diabetes/ -p ../../Parameters/sssom_0"
tmux new-session -d "conda activate deepunsupervised & python train_sssom.py --batch-size 32 --start-idx 10 --stop-idx 20 -i ../../Parameters/batch_sssom_uci/diabetes/inputPathsReal_train_diabetes -t ../../Parameters/batch_sssom_uci/diabetes/inputPathsReal_test_diabetes -r ../../Parameters/batch_sssom_uci/diabetes/ -p ../../Parameters/sssom_0"
tmux new-session -d "conda activate deepunsupervised & python train_sssom.py --batch-size 32 --start-idx 20 --stop-idx 30 -i ../../Parameters/batch_sssom_uci/diabetes/inputPathsReal_train_diabetes -t ../../Parameters/batch_sssom_uci/diabetes/inputPathsReal_test_diabetes -r ../../Parameters/batch_sssom_uci/diabetes/ -p ../../Parameters/sssom_0"
tmux new-session -d "conda activate deepunsupervised & python train_sssom.py --batch-size 32 --start-idx 30 --stop-idx 40 -i ../../Parameters/batch_sssom_uci/diabetes/inputPathsReal_train_diabetes -t ../../Parameters/batch_sssom_uci/diabetes/inputPathsReal_test_diabetes -r ../../Parameters/batch_sssom_uci/diabetes/ -p ../../Parameters/sssom_0"
tmux new-session -d "conda activate deepunsupervised & python train_sssom.py --batch-size 32 --start-idx 40 --stop-idx 50 -i ../../Parameters/batch_sssom_uci/diabetes/inputPathsReal_train_diabetes -t ../../Parameters/batch_sssom_uci/diabetes/inputPathsReal_test_diabetes -r ../../Parameters/batch_sssom_uci/diabetes/ -p ../../Parameters/sssom_0"