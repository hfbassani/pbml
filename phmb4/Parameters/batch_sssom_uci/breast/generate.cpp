#include <bits/stdc++.h>
using namespace std;


int main()
{
	
	for(int i = 0 ; i < 500 ; i+=10)
	{
		cout << "python train_sssom.py --batch-size 32 --start-idx " << i << " --stop-idx " << i+10 << " -i ../../Parameters/batch_sssom_uci/breast/inputPathsReal_train_breast -t ../../Parameters/batch_sssom_uci/breast/inputPathsReal_test_breast -r ../../Parameters/batch_sssom_uci/breast/ -p ../../Parameters/sssom_0" << endl;
	}
	

	return 0;
}



