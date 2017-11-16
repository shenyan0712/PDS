#include <math.h>

#include "spdsys.h"

bool dpotrs_v1_p1(clBufferEx<double> &L_buf, clBufferEx<double> &x_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
	int matSize, int tileSize, cldev &cd, int devIdx);
bool dpotrs_v1_p2(clBufferEx<double> &L_buf, clBufferEx<double> &x_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
	int matSize, int tileSize, cldev &cd, int devIdx);


/*
OpenCL 1.2
求解对称正定系统Ax=b
L	==> 对A进行cholesky分解得到的下三角矩阵
DBI	==> 对A进行cholesky分解时产生的对角块Ljj的逆
*/
bool dpotrs_v1(clBufferEx<double> &L_buf, clBufferEx<double> &x_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
                  int matSize, int tileSize, cldev &cd, int devIdx)
{
    //定义一个buffer用于存放中间结果c
    clBufferEx<double> c_buf = clBufferEx<double>(cd.get_context(), cd.get_queue(devIdx), b_buf.size(), MODE_COARSE_SVM);

    dpotrs_v1_p1(L_buf, c_buf, b_buf, DBIs_buf, matSize, tileSize, cd, devIdx);

    dpotrs_v1_p2(L_buf, x_buf, c_buf, DBIs_buf, matSize, tileSize, cd, devIdx);

    return true;
}



/*
求解Lx=b,
L	==> 下三角矩阵，
DBIs	==> L的对角块Ljj的逆
*/
bool dpotrs_v1_p1(clBufferEx<double> &L_buf, clBufferEx<double> &x_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
                        int matSize, int tileSize, cldev &cd, int devIdx)
{
    int num;
    cl_int err;
    int nblks = matSize / tileSize;
    int NWR = 4;		//the number of workgroup rows,2的倍数

    cl::Kernel *kernel_step1 = cd.get_kernel("dpotrs_v1_p1_s1");
    cl::Kernel *kernel_step2 = cd.get_kernel("dpotrs_v1_p1_s2");
    cl::Kernel *kernel_step3 = cd.get_kernel("dpotrs_v1_p1_s3");

    cl::Device dev = cd.get_device(devIdx);

    for (int i = 0; i < nblks; i++)
    {
        //***** step 1.1.
        if (i > 0)
        {
            num = 0;
            err = kernel_step1->setArg(num++, i);			// first column
            err = kernel_step1->setArg(num++, NWR);
            err = kernel_step1->setArg(num++, tileSize);
            err = kernel_step1->setArg(num++, matSize);
            err = kernel_step1->setArg(num++, nblks);
            err = L_buf.SetArgForKernel(*kernel_step1, num++);
            err = DBIs_buf.SetArgForKernel(*kernel_step1, num++);
            err = b_buf.SetArgForKernel(*kernel_step1, num++);
            err = x_buf.SetArgForKernel(*kernel_step1, num++);
            err = cd.get_queue(devIdx).enqueueNDRangeKernel(
                      *kernel_step1, cl::NullRange, cl::NDRange(i *tileSize),		//globalWorkSize
                      cl::NDRange(tileSize), NULL, NULL); //cl::NDRange(WI_SIZE)
            cd.get_queue(devIdx).finish();
        }

        //***** step 1.2
        int nCols = i;
        int loop = 0;
        int nrb = i;	//实际需要累加的行
        while (nCols > 1)
        {
            int g_stride = pow(NWR, loop);						//g_stride应该是NWR*2^i，i为当前的loop
            nCols = (int)NWR*ceil((double)nCols / NWR);			//
            num = 0;
            err = kernel_step2->setArg(num++, i);			//
            err = kernel_step2->setArg(num++, NWR);
            err = kernel_step2->setArg(num++, nrb);
            err = kernel_step2->setArg(num++, nCols);
            err = kernel_step2->setArg(num++, loop);
            err = kernel_step2->setArg(num++, tileSize);
            err = kernel_step2->setArg(num++, matSize);
            err = kernel_step2->setArg(num++, nblks);
            err = kernel_step2->setArg(num++, g_stride);
            err = L_buf.SetArgForKernel(*kernel_step2, num++);
            err = DBIs_buf.SetArgForKernel(*kernel_step2, num++);
            err = b_buf.SetArgForKernel(*kernel_step2, num++);
            err = x_buf.SetArgForKernel(*kernel_step2, num++);

            err = cd.get_queue(devIdx).enqueueNDRangeKernel(
                      *kernel_step2, cl::NullRange, cl::NDRange(nCols),		//globalWorkSize
                      cl::NDRange(NWR), NULL, NULL); //cl::NDRange(WI_SIZE)
            cd.get_queue(devIdx).finish();

            loop++;
            nCols = nCols / NWR;
        }

        //***** step 1.3
        num = 0;
        err = kernel_step3->setArg(num++, i);			// first column
        err = kernel_step3->setArg(num++, NWR);
        err = kernel_step3->setArg(num++, tileSize);
        err = kernel_step3->setArg(num++, matSize);
        err = kernel_step3->setArg(num++, nblks);
        err = L_buf.SetArgForKernel(*kernel_step3, num++);
        err = DBIs_buf.SetArgForKernel(*kernel_step3, num++);
        err = b_buf.SetArgForKernel(*kernel_step3, num++);
        err = x_buf.SetArgForKernel(*kernel_step3, num++);
        err = kernel_step3->setArg(num++, tileSize * sizeof(double), NULL);
        err = cd.get_queue(devIdx).enqueueNDRangeKernel(
                  *kernel_step3, cl::NullRange, cl::NDRange(tileSize),		//globalWorkSize
                  cl::NDRange(tileSize), NULL, NULL);
        cd.get_queue(devIdx).finish();
    }

    return false;
}


/*
/*
求解Lt*x=b,
L	==> 下三角矩阵，但注意需要的是Lt
DBIs	==> L的对角块Ljj的逆inv(Ljj), 但注意需要的是inv(Ljj)的转置
*/
bool dpotrs_v1_p2(clBufferEx<double> &L_buf, clBufferEx<double> &x_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
                        int matSize, int tileSize, cldev &cd, int devIdx)
{
    int num;
    cl_int err;
    int nblks = matSize / tileSize;
    int NWR = 4;		//the number of workgroup rows,2的倍数

    cl::Kernel *kernel_step1 = cd.get_kernel("dpotrs_v1_p2_s1");
    cl::Kernel *kernel_step2 = cd.get_kernel("dpotrs_v1_p2_s2");
    cl::Kernel *kernel_step3 = cd.get_kernel("dpotrs_v1_p2_s3");

    cl::Device dev = cd.get_device(devIdx);

    for (int i = nblks-1; i >=0; i--)
    {
        //***** step 1.1.
        if (i < nblks-1)
        {
            num = 0;
            err = kernel_step1->setArg(num++, i);			//
            err = kernel_step1->setArg(num++, NWR);
            err = kernel_step1->setArg(num++, tileSize);
            err = kernel_step1->setArg(num++, matSize);
            err = kernel_step1->setArg(num++, nblks);
            err = L_buf.SetArgForKernel(*kernel_step1, num++);
            err = DBIs_buf.SetArgForKernel(*kernel_step1, num++);
            err = b_buf.SetArgForKernel(*kernel_step1, num++);
            err = x_buf.SetArgForKernel(*kernel_step1, num++);
            err = cd.get_queue(devIdx).enqueueNDRangeKernel(
                      *kernel_step1, cl::NullRange, cl::NDRange((nblks - i-1) *tileSize),		//globalWorkSize
                      cl::NDRange(tileSize), NULL, NULL);										//cl::NDRange(WI_SIZE)
            cd.get_queue(devIdx).finish();
        }

        //***** step 1.2
        int nrb = nblks - i - 1;	//实际需要累加的行
        int nCols = nrb;
        int loop = 0;
        while (nCols > 1)
        {
            int g_stride = pow(NWR, loop);						//g_stride应该是NWR*2^i，i为当前的loop
            nCols = (int)NWR*ceil((double)nCols / NWR);			//
            num = 0;
            err = kernel_step2->setArg(num++, i);			// first column
            err = kernel_step2->setArg(num++, NWR);
            err = kernel_step2->setArg(num++, nrb);
            err = kernel_step2->setArg(num++, nCols);
            err = kernel_step2->setArg(num++, loop);
            err = kernel_step2->setArg(num++, tileSize);
            err = kernel_step2->setArg(num++, matSize);
            err = kernel_step2->setArg(num++, nblks);
            err = kernel_step2->setArg(num++, g_stride);
            err = L_buf.SetArgForKernel(*kernel_step2, num++);
            err = DBIs_buf.SetArgForKernel(*kernel_step2, num++);
            err = b_buf.SetArgForKernel(*kernel_step2, num++);
            err = x_buf.SetArgForKernel(*kernel_step2, num++);

            err = cd.get_queue(devIdx).enqueueNDRangeKernel(
                      *kernel_step2, cl::NullRange, cl::NDRange(nCols),		//globalWorkSize
                      cl::NDRange(NWR), NULL, NULL); //cl::NDRange(WI_SIZE)
            cd.get_queue(devIdx).finish();
            loop++;
            nCols = nCols / NWR;
        }

        //***** step 1.3
        num = 0;
        err = kernel_step3->setArg(num++, i);			// first column
        err = kernel_step3->setArg(num++, NWR);
        err = kernel_step3->setArg(num++, tileSize);
        err = kernel_step3->setArg(num++, matSize);
        err = kernel_step3->setArg(num++, nblks);
        err = L_buf.SetArgForKernel(*kernel_step3, num++);
        err = DBIs_buf.SetArgForKernel(*kernel_step3, num++);
        err = b_buf.SetArgForKernel(*kernel_step3, num++);
        err = x_buf.SetArgForKernel(*kernel_step3, num++);
        err = kernel_step3->setArg(num++, tileSize * sizeof(double), NULL);
        err = cd.get_queue(devIdx).enqueueNDRangeKernel(
                  *kernel_step3, cl::NullRange, cl::NDRange(tileSize),		//globalWorkSize
                  cl::NDRange(tileSize), NULL, NULL);
        cd.get_queue(devIdx).finish();

    }

    return false;
}
