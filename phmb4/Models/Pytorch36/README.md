CUDA_LAUNCH_BLOCKING=1 python -m cProfile -o output.prof train_sssom.py -i ../../Parameters/inputPath_mnist -t ../../Parameters/inputPath_mnist -r mnist/ -p ../../Parameters/deep_mnist10 --batch-size 32 --cuda



CUDA_LAUNCH_BLOCKING=1 python -m memory_profiler -o output.prof train_sssom.py -i ../../Parameters/inputPath_mnist -t ../../Parameters/inputPath_mnist -r mnist/ -p ../../Parameters/deep_mnist10 --batch-size 32 --cuda



watch -n 2 nvidia-smi


nakeviz output.prof 

py-spy -- python train_sssom.py -i ../../Parameters/inputPath_mnist -t ../../Parameters/inputPath_mnist -r mnist/ -p ../../Parameters/deep_mnist10 --batch-size 32 --cuda