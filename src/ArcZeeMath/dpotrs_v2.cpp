#include <math.h>

#include "spdsys.h"

bool dpotrs_v2_p1(clBufferEx<double> &L_buf, clBufferEx<double> &x_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
	int matSize, int tileSize, cldev &cd, int devIdx);
bool dpotrs_v2_p2(clBufferEx<double> &L_buf, clBufferEx<double> &x_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
	int matSize, int tileSize, cldev &cd, int devIdx);


/*
OpenCL 1.2
求解对称正定系统Ax=b
L	==> 对A进行cholesky分解得到的下三角矩阵
DBI	==> 对A进行cholesky分解时产生的对角块Ljj的逆
*/
bool dpotrs_v2(clBufferEx<double> &L_buf, clBufferEx<double> &x_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
                  int matSize, int tileSize, cldev &cd, int devIdx)
{
    //定义一个buffer用于存放中间结果c
    clBufferEx<double> c_buf = clBufferEx<double>(cd.get_context(), cd.get_queue(devIdx), b_buf.size(), MODE_COARSE_SVM);

    dpotrs_v2_p1(L_buf, c_buf, b_buf, DBIs_buf, matSize, tileSize, cd, devIdx);

    dpotrs_v2_p2(L_buf, x_buf, c_buf, DBIs_buf, matSize, tileSize, cd, devIdx);

    return true;
}



/*
求解Lx=b,
L	==> 下三角矩阵，
DBIs	==> L的对角块Ljj的逆
*/
bool dpotrs_v2_p1(clBufferEx<double> &L_buf, clBufferEx<double> &x_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
                        int matSize, int tileSize, cldev &cd, int devIdx)
{
    int num;
    cl_int err;
    int nblks = matSize / tileSize;

    cl::Kernel *kernel_step1 = cd.get_kernel("dpotrs_v2_p1");
    cl::Device dev = cd.get_device(devIdx);

	num = 1;
	err = kernel_step1->setArg(num++, matSize);
	err = kernel_step1->setArg(num++, nblks);
	err = L_buf.SetArgForKernel(*kernel_step1, num++);
	err = DBIs_buf.SetArgForKernel(*kernel_step1, num++);
	err = b_buf.SetArgForKernel(*kernel_step1, num++);
	err = x_buf.SetArgForKernel(*kernel_step1, num++);

    for (int i = 0; i < nblks; i++)
    {
        num = 0;
        err = kernel_step1->setArg(0, i);			// first column
        err = cd.get_queue(devIdx).enqueueNDRangeKernel(
                    *kernel_step1, cl::NullRange, cl::NDRange(tileSize),		//globalWorkSize
                    cl::NDRange(tileSize), NULL, NULL); //cl::NDRange(WI_SIZE)
		cd.get_queue(devIdx).flush();

		//if (i == 1) return true;
    }
	cd.get_queue(devIdx).finish();
    return true;
}


/*
/*
求解Lt*x=b,
L	==> 下三角矩阵，但注意需要的是Lt
DBIs	==> L的对角块Ljj的逆inv(Ljj), 但注意需要的是inv(Ljj)的转置
*/
bool dpotrs_v2_p2(clBufferEx<double> &L_buf, clBufferEx<double> &x_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
                        int matSize, int tileSize, cldev &cd, int devIdx)
{
    int num;
    cl_int err;
    int nblks = matSize / tileSize;
    int NWR = 4;		//the number of workgroup rows,2的倍数

    cl::Kernel *kernel_step1 = cd.get_kernel("dpotrs_v2_p2");
    cl::Device dev = cd.get_device(devIdx);

	num = 1;
	err = kernel_step1->setArg(num++, matSize);
	err = kernel_step1->setArg(num++, nblks);
	err = L_buf.SetArgForKernel(*kernel_step1, num++);
	err = DBIs_buf.SetArgForKernel(*kernel_step1, num++);
	err = b_buf.SetArgForKernel(*kernel_step1, num++);
	err = x_buf.SetArgForKernel(*kernel_step1, num++);

    for (int i = nblks-1; i >=0; i--)
    {
            num = 0;
            err = kernel_step1->setArg(0, i);			//
            err = cd.get_queue(devIdx).enqueueNDRangeKernel(
                      *kernel_step1, cl::NullRange, cl::NDRange(tileSize),		//globalWorkSize
                      cl::NDRange(tileSize), NULL, NULL);										//cl::NDRange(WI_SIZE)
			cd.get_queue(devIdx).flush();

			//if (i == 1) return true;
    }
	cd.get_queue(devIdx).finish();
    return false;
}
