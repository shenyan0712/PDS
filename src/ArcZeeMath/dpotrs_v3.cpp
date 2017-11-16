#include <math.h>

#include "spdsys.h"

bool dpotrs_v3_p1(clBufferEx<double> &L_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
	int matSize, int tileSize, cldev &cd, int devIdx);
bool dpotrs_v3_p2(clBufferEx<double> &L_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
	int matSize, int tileSize, cldev &cd, int devIdx);


/*
OpenCL 1.2
求解对称正定系统Ax=b
L	==> 对A进行cholesky分解得到的下三角矩阵
DBI	==> 对A进行cholesky分解时产生的对角块Ljj的逆
*/
bool dpotrs_v3(clBufferEx<double> &L_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
                  int matSize, int tileSize, cldev &cd, int devIdx)
{
    //定义一个buffer用于存放中间结果c
    //clBufferEx<double> c_buf = clBufferEx<double>(cd.get_context(), cd.get_queue(devIdx), b_buf.size(), MODE_COARSE_SVM);

    dpotrs_v3_p1(L_buf, b_buf, DBIs_buf, matSize, tileSize, cd, devIdx);

    dpotrs_v3_p2(L_buf, b_buf, DBIs_buf, matSize, tileSize, cd, devIdx);

    return true;
}



/*
求解Lx=b,
L	==> 下三角矩阵，
DBIs	==> L的对角块Ljj的逆
*/
bool dpotrs_v3_p1(clBufferEx<double> &L_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
                        int matSize, int tileSize, cldev &cd, int devIdx)
{
    int num;
    cl_int err;
    int nblks = matSize / tileSize;

    cl::Kernel *kernel_p1_s1 = cd.get_kernel("dpotrs_v3_p1_s1");
	cl::Kernel *kernel_p1_s2 = cd.get_kernel("dpotrs_v3_p1_s2");
    cl::Device dev = cd.get_device(devIdx);

	num = 1;
	err = kernel_p1_s1->setArg(num++, matSize);
	err = L_buf.SetArgForKernel(*kernel_p1_s1, num++);
	err = DBIs_buf.SetArgForKernel(*kernel_p1_s1, num++);
	err = b_buf.SetArgForKernel(*kernel_p1_s1, num++);

	num = 1;
	err = kernel_p1_s2->setArg(num++, matSize);
	err = L_buf.SetArgForKernel(*kernel_p1_s2, num++);
	err = DBIs_buf.SetArgForKernel(*kernel_p1_s2, num++);
	err = b_buf.SetArgForKernel(*kernel_p1_s2, num++);

    for (int i = 0; i < nblks; i++)
    {
        err = kernel_p1_s1->setArg(0, i);
        err = cd.get_queue(devIdx).enqueueNDRangeKernel(
                    *kernel_p1_s1, cl::NullRange, cl::NDRange(tileSize),		//globalWorkSize
                    cl::NDRange(tileSize), NULL, NULL); //cl::NDRange(WI_SIZE)
		cd.get_queue(devIdx).flush();

		//if (i == 1) return true;

		if (i == nblks - 1) break;
		err = kernel_p1_s2->setArg(0, i);
		err = cd.get_queue(devIdx).enqueueNDRangeKernel(
			*kernel_p1_s2, cl::NullRange, cl::NDRange(matSize-(i+1)*tileSize),		//globalWorkSize
			cl::NDRange(tileSize), NULL, NULL); //cl::NDRange(WI_SIZE)
		cd.get_queue(devIdx).flush();

		//if (i == 0) return true;
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
bool dpotrs_v3_p2(clBufferEx<double> &L_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
                        int matSize, int tileSize, cldev &cd, int devIdx)
{
	int num;
	cl_int err;
	int nblks = matSize / tileSize;

	cl::Kernel *kernel_p2_s1 = cd.get_kernel("dpotrs_v3_p2_s1");
	cl::Kernel *kernel_p2_s2 = cd.get_kernel("dpotrs_v3_p2_s2");
	cl::Device dev = cd.get_device(devIdx);

	num = 1;
	err = kernel_p2_s1->setArg(num++, matSize);
	err = L_buf.SetArgForKernel(*kernel_p2_s1, num++);
	err = DBIs_buf.SetArgForKernel(*kernel_p2_s1, num++);
	err = b_buf.SetArgForKernel(*kernel_p2_s1, num++);

	num = 1;
	err = kernel_p2_s2->setArg(num++, matSize);
	err = L_buf.SetArgForKernel(*kernel_p2_s2, num++);
	err = DBIs_buf.SetArgForKernel(*kernel_p2_s2, num++);
	err = b_buf.SetArgForKernel(*kernel_p2_s2, num++);

	for (int i = nblks-1; i >=0; i--)
	{
		err = kernel_p2_s1->setArg(0, i);
		err = cd.get_queue(devIdx).enqueueNDRangeKernel(
			*kernel_p2_s1, cl::NullRange, cl::NDRange(tileSize),		//globalWorkSize
			cl::NDRange(tileSize), NULL, NULL); //cl::NDRange(WI_SIZE)
		cd.get_queue(devIdx).flush();

		//if (i == nblks - 2) break;

		if (i == 0) break;
		err = kernel_p2_s2->setArg(0, i);
		err = cd.get_queue(devIdx).enqueueNDRangeKernel(
			*kernel_p2_s2, cl::NullRange, cl::NDRange( i*tileSize),		//globalWorkSize
			cl::NDRange(tileSize), NULL, NULL); //cl::NDRange(WI_SIZE)
		cd.get_queue(devIdx).flush();

		//if (i == nblks - 1) break;
	}
	cd.get_queue(devIdx).finish();
	return true;
}
