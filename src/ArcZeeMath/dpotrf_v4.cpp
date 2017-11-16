#include <math.h>

#include "spdsys.h"
#include "cldef.h"

/*
A_buf	==>nxn的对称矩阵, row-major storage
DBI_buf ==>diagonal block inverse
如果A不能cholesky分解，则A[0,0]=-1.0
采用opencl 1.2的方式
*/
bool dpotrf_v4_cl(clBufferEx<double> &A_buf, int matSize, int tileSize, cldev &cd, int devIdx, clBufferEx<double> &DBIs_buf)
{
	double ret;
	cl_int err;
	int num = 0;
	int cols = (int)ceil(matSize / tileSize);

	cl::Kernel *kernel_step1 = cd.get_kernel("chol_v4_step1");
	cl::Kernel *kernel_step2 = cd.get_kernel("chol_v4_step2");
	cl::Kernel *kernel_step3 = cd.get_kernel("chol_v4_step3");
	cl::Device dev = cd.get_device(devIdx);
	clBufferEx<double> ret_buf = clBufferEx<double>(cd.get_context(), cd.get_queue(devIdx), 1, MODE_NO_SVM);

	int TN = tileSize / VEC_WIDTH;
	int MN = matSize / VEC_WIDTH;	//

	num = 3;
	err = kernel_step1->setArg(num++, MN);
	err = A_buf.SetArgForKernel(*kernel_step1, num++);
	err = DBIs_buf.SetArgForKernel(*kernel_step1, num++);
	err = ret_buf.SetArgForKernel(*kernel_step1, num);

	num = 1;
	err = kernel_step2->setArg(num++, tileSize);
	err = kernel_step2->setArg(num++, matSize);
	err = A_buf.SetArgForKernel(*kernel_step2, num++);
	err = DBIs_buf.SetArgForKernel(*kernel_step2, num++);
	err = ret_buf.SetArgForKernel(*kernel_step2, num++);

	num = 1;
	err = kernel_step3->setArg(num++, tileSize);
	err = kernel_step3->setArg(num++, matSize);
	err = kernel_step3->setArg(num++, TN);
	err = kernel_step3->setArg(num++, MN);
	err = A_buf.SetArgForKernel(*kernel_step3, num++);
	err = DBIs_buf.SetArgForKernel(*kernel_step3, num++);
	err = ret_buf.SetArgForKernel(*kernel_step3, num++);

	for (int j = 0; j < cols; j++)
	{
		//call kernel step 1. compute T_cdj=A_cdj-L_abj*Lt_abj
		err = kernel_step1->setArg(0, j);
		err = kernel_step1->setArg(1, j*TileSize);
		err = kernel_step1->setArg(2, j*TileSize_C);
		cd.get_queue(0).enqueueNDRangeKernel(
			*kernel_step1, cl::NullRange, cl::NDRange(matSize - j*TileSize, TileSize_C),		//globalWorkSize
			cl::NDRange(TileSize_R, TileSize_C), NULL, NULL); //cl::NDRange(WI_SIZE)
		cd.get_queue(0).flush();
		//if (j == 1) return true;

		//call kernel step 2. compute L_jj and inv(L_jj)
		err = kernel_step2->setArg(0, j);
		cd.get_queue(0).enqueueNDRangeKernel(
			*kernel_step2, cl::NullRange, cl::NDRange(tileSize, tileSize),		//globalWorkSize
			cl::NDRange(tileSize, tileSize), NULL, NULL); //cl::NDRange(WI_SIZE)
		cd.get_queue(0).flush();
		//if (j == 1) return true;

		//call kernel step 3. compute L_dj=T_dj*inv(L_jj)
		if (j == cols - 1) continue; //最后一个块是对角块
		err = kernel_step3->setArg(0, j);
		cd.get_queue(0).enqueueNDRangeKernel(
			*kernel_step3, cl::NullRange, cl::NDRange(matSize - (j + 1)*tileSize, tileSize / VEC_WIDTH),		//globalWorkSize
			cl::NDRange(tileSize, tileSize / VEC_WIDTH), NULL, NULL);
		cd.get_queue(0).flush();//cl::NDRange(WI_SIZE)
								//if (j == 1) return true;
	}
	cd.get_queue(0).finish();
	return true;
}
