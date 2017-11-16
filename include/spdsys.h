#pragma once

#include <cldev.h>
/*
对称正定系统的求解
*/
#include <clBufferEx.h>

#define FMULS_POTRF(n_) ((n_) * (((1. / 6.) * (n_) + 0.5) * (n_) + (1. / 3.)))
#define FADDS_POTRF(n_) ((n_) * (((1. / 6.) * (n_)      ) * (n_) - (1. / 6.)))

#define FMULS_POTRI(n_) ( (n_) * ((2. / 3.) + (n_) * ((1. / 3.) * (n_) + 1. )) )
#define FADDS_POTRI(n_) ( (n_) * ((1. / 6.) + (n_) * ((1. / 3.) * (n_) - 0.5)) )

#define FMULS_POTRS(n_, nrhs_) ((nrhs_) * (n_) * ((n_) + 1 ))
#define FADDS_POTRS(n_, nrhs_) ((nrhs_) * (n_) * ((n_) - 1 ))



/*
单个OpenCL设备的Cholesky分解
A	==> nxn matrix
n	==> dimension of A
cd  ==> 
DBI	==> inverse of Ljj blocks
*/
bool dpotrf_v1_cl(clBufferEx<double> &A_buf, int matSize, int tileSize, cldev &cd, int devIdx, clBufferEx<double> &DBI_buf);
bool dpotrf_v2_cl(clBufferEx<double> &A_buf, int matSize, int tileSize, cldev &cd, int devIdx, clBufferEx<double> &DBIs_buf);
bool dpotrf_v3_cl(clBufferEx<double> &A_buf, int matSize, int tileSize, cldev &cd, int devIdx, clBufferEx<double> &DBIs_buf);
bool dpotrf_v4_cl(clBufferEx<double> &A_buf, int matSize, int tileSize, cldev &cd, int devIdx, clBufferEx<double> &DBIs_buf);
//bool dpotrf_v5_cl(clBufferEx<double> &A_buf, int matSize, int tileSize, cldev &cd, int devIdx, clBufferEx<double> &DBIs_buf);


/*
求解对称正定系统Ax=b
L	==> 对A进行cholesky分解得到的下三角矩阵，A=L*Lt
DBI	==> 对A进行cholesky分解时产生的对角块Ljj的逆
*/
bool dpotrs_v1(clBufferEx<double> &L_buf, clBufferEx<double> &x_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBI_buf,
	int matSize, int tileSize, cldev &cd, int devIdx);
bool dpotrs_v2(clBufferEx<double> &L_buf, clBufferEx<double> &x_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBI_buf,
	int matSize, int tileSize, cldev &cd, int devIdx);

bool dpotrs_v3(clBufferEx<double> &L_buf, clBufferEx<double> &b_buf, clBufferEx<double> &DBIs_buf,
	int matSize, int tileSize, cldev &cd, int devIdx);