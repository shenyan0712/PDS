///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//使用local memory优化

/*
step 1. 计算T_cdj=A_cdj-L_abj*Lt_aj.
*/
__kernel void chol_v2_step1(
	const int use_devqueue,
	const int j,			//当前处理的列
	const int tileSize,
	const int matSize,
	__global double *A,
	__global double *DBIs,
	__global double *ret,
	__local double *TA,
	__local double *TB
)
{
	int bi = get_group_id(0)+j;
	int u = get_local_id(0);
	int v = get_local_id(1);
	int uvAddr = u*tileSize + v;

	int bandSize = tileSize*matSize;
	int uSize = u*matSize;
	int addr1 = uSize + v;
	int addr2 = bi*bandSize;

	double sum = 0;

	//每一次提取一个块的内容到TA和 TB
	for (int n = 0; n < j; n++)
	{
		int nSize = n*tileSize;
		TA[uvAddr] = A[addr2 + nSize+addr1];	//第bi行的第n个块
		TB[uvAddr] = A[j*bandSize + nSize+ addr1];	//恒为第j行
		barrier(CLK_LOCAL_MEM_FENCE);	//等待块加载完
		//计算其中的
		for (int k = 0; k < tileSize; k++)
			sum += TA[u*tileSize + k] * TB[v*tileSize + k];
		barrier(CLK_LOCAL_MEM_FENCE);	//等待块计算完
	}

	A[addr2 + uSize + j*tileSize + v] -= sum;

	//OpenCL 2.0, device queue
#ifdef ENABLE_OPENCL_2_0
	/*
	if (x==0 && y==0 && use_devqueue && A[0]>0)  //j<(cols-1)确保最后一列不会执行chol_step2
	{
		//call chol_step2
		void(^kern_wrapper)() = ^() {
			chol_v2_step2(1, j, tileSize, matSize, A, DBIs,ret);
		};
		size_t    global_size[2] = { (matSize/tileSize-j-1)*tileSize, tileSize };
		size_t    local_size[2] = { tileSize,tileSize };
		ndrange_t ndrange = ndrange_2D(global_size, local_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper
		);
	}
	*/
#endif
}



/*
计算第j列对角块的L_jj, inv(L_jj）
*/
__kernel void chol_v2_step2(
	const int use_devqueue,
	const int j,			//当前处理的列
	const int tileSize,
	const int matSize,
	__global double *A,
	__global double *DBIs,
	__global double *ret,
	__local double *DB,		//tileSize*tileSize, 用于存储
	__local double *DBI		//tileSize*tileSize
	)
{
	int flag;
	int S = j*tileSize;
	int u = get_local_id(0);
	int v = get_local_id(1);

	int x = u + S;
	int y = v + S;
	int uvAddr = u*tileSize + v;
	int xyAddr = x*matSize + y;

	DB[uvAddr] = A[xyAddr];
	barrier(CLK_LOCAL_MEM_FENCE);

	flag = compute_Ljj(DB, uvAddr, tileSize, u, v);
	compute_LjjInv(DB, DBI, uvAddr, tileSize, u, v);

	A[xyAddr] = DB[uvAddr];
	DBIs[S*tileSize + uvAddr] = DBI[uvAddr];

	//OpenCL 2.0, device queue
#ifdef ENABLE_OPENCL_2_0
	/*
	int cols = matSize / tileSize;
	if (get_global_id(0) == 0 && get_global_id(1) == 0 && j<(cols-1) && use_devqueue)
	{
		//call chol_step1 with j+1;
		void(^kern_wrapper)(local void *, local void *) = ^ (local void *B, local void *DBI) {
			chol_v2_step1(1, j+1, tileSize, matSize, A, DBIs, ret, B, DBI);
		};
		size_t    global_size[2] = { matSize, tileSize };
		size_t    local_size[2] = { tileSize,tileSize };
		ndrange_t ndrange = ndrange_2D(global_size, local_size);
		enqueue_kernel(
			get_default_queue(),
			CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
			ndrange, kern_wrapper,
			(unsigned int)(tileSize * tileSize * sizeof(double)),		//size of local memory B
			(unsigned int)(tileSize * tileSize * sizeof(double))		//size of local memory DBI
		);
	}
	*/
#endif
}


/*
使用inv(L_jj)计算剩余L_dj=T_dj*inv(L_jj)
*/
__kernel void chol_v2_step3(
	const int use_devqueue,
	const int j,			//当前处理的列
	const int tileSize,
	const int matSize,
	__global double *A,
	__global double *DBIs,
	__global double *ret,
	__local double *T,
	__local double *DBI
)
{
	int S = j*tileSize;
	int u = get_local_id(0);
	int v = get_local_id(1);
	int x = get_global_id(0) + S+tileSize;
	int y = get_global_id(1) + S;

	int uvAddr = u*tileSize + v;
	int xyAddr = x*matSize + y;

	//copy tileSize*tileSize block of A to T
	T[uvAddr] = A[xyAddr];
	DBI[uvAddr] = DBIs[S*tileSize + uvAddr];
	barrier(CLK_LOCAL_MEM_FENCE);

	//T_dj的u行[j*tileSize, (j+1)*tileSize)和inv(L_jj)的v行的点积
	double sum = 0;
	for (int k = 0; k < tileSize; k++) {
		sum += T[u*tileSize + k] * DBI[v*tileSize + k];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	A[x*matSize + y] = sum; // DBIs[vAddr + 1];
}