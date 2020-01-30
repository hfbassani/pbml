#include <bits/stdc++.h>
using namespace std;


int main()
{
	
	for(int i = 0 ; i < 500 ; i+=10)
	{
		cout << "tmux new-session -d \"conda activate deepunsupervised & python train_sssom.py --batch-size 32 --start-idx " << i << " --stop-idx " << i+10 << " -i ../../Parameters/batch_sssom_uci/diabetes/inputPathsReal_train_diabetes -t ../../Parameters/batch_sssom_uci/diabetes/inputPathsReal_test_diabetes -r ../../Parameters/batch_sssom_uci/diabetes/ -p ../../Parameters/sssom_0\"" << endl;

	}
	

	return 0;
}



