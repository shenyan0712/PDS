
#include<iostream>

#include<cldev.h>

using namespace std;

int main()
{
    cout<<"hello world"<<endl;

	//==========openCL initialization
	cldev cd;
	cd.init(true);
	//selecte devices
	if (cd.selectPfWithMostDev(CL_DEVICE_TYPE_GPU, 1.2, 0)) {
		cout << "No devices satisfy the requirement.";
	}

    system("pause");

    return 0;
}
