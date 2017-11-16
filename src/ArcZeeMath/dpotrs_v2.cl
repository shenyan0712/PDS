

/*
Step1 
*/
__kernel void dpotrs_v2_p1(
	const int i,			//当前处理的行
	const int matSize,
	const int n,			//块矩阵的尺寸， n=matSize/tileSize
	__global double *L,
	__global double *DBIs,
	__global double *b,		//Lx=b
	__global double *x		//results
)
{
	const int lid = get_local_id(0);
	const int base_addr = (i*TileSize+lid)*matSize;			//该工作项读取Lik的第lid行

	local double T[TileSize][TileSize];
	local double total_sum[TileSize];
	double x_k[TileSize];
	double sum=0;
	const int i_TileSize = i*TileSize;

	//计算t_i=b_i-sum(Lin*xn)
	for (int k = 0; k < i; k++) {
		//读取Lin块到T
		int k_TileSize = k*TileSize;
		int addr = base_addr + k_TileSize;
		for (int t = 0; t < TileSize; t++)
			T[lid][t] = L[addr+t];
		barrier(CLK_LOCAL_MEM_FENCE);

		//读取x_k
		for (int t = 0; t < TileSize; t++)
			x_k[t] = x[k_TileSize+t];

		//计算Lin的第lid行与x_n的点积
		for (int t = 0; t < TileSize; t++)
			sum += T[lid][t] * x_k[t];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	total_sum[lid] =b[i_TileSize+lid]-sum;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//计算inv(Lii)*t_i
	//读取inv(Lii)块到T
	for (int t = 0; t < TileSize; t++)
		T[lid][t] = DBIs[i_TileSize*TileSize + lid*TileSize + t];

	barrier(CLK_LOCAL_MEM_FENCE);

	sum = 0;
	for (int t = 0; t < TileSize; t++)
		sum += T[lid][t] * total_sum[t];

	x[i_TileSize + lid] = sum;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
Step 2. 
*/
__kernel void dpotrs_v2_p2(
	const int i,			//当前处理的行
	const int matSize,
	const int n,			//块矩阵的尺寸，n=matSize/tileSize
	__global double *L,
	__global double *DBIs,
	__global double *b,		//Lx=b
	__global double *x		//results
)
{
	const int lid = get_local_id(0);							//该工作项计算x_i的第lid个元素
	local double T[TileSize][TileSize];
	local double total_sum[TileSize];
	double x_k[TileSize];
	double sum = 0;
	const int i_TileSize = i*TileSize;

	//计算t_i=b_i-sum(Lki*xk)
	for (int k = i+1; k< n; k++) {
		int k_TileSize = k*TileSize;
		//读取Lki的第lid行, 存入T的第lid列(即将Lik的转置)
		int addr = (k_TileSize+lid)*matSize + i_TileSize;
		for (int t = 0; t < TileSize; t++)
			T[t][lid] = L[addr+t];
		barrier(CLK_LOCAL_MEM_FENCE);

		//读取x_k
		for (int t = 0; t < TileSize; t++)
			x_k[t] = x[k_TileSize+t];

		barrier(CLK_LOCAL_MEM_FENCE);
		//计算Lki的第lid列与x_k的点积（也即T的第lid行与x_k的点积）
		for (int t = 0; t < TileSize; t++)
			sum += T[lid][t] * x_k[t];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	total_sum[lid] = b[i_TileSize + lid] - sum;

	barrier(CLK_LOCAL_MEM_FENCE);

	//计算inv(Lii)*t_i
	//读取inv(Lii)块到T
	for (int t = 0; t < TileSize; t++)
		T[t][lid] = DBIs[i_TileSize*TileSize + lid*TileSize + t];

	barrier(CLK_LOCAL_MEM_FENCE);

	sum = 0;
	for (int t = 0; t < TileSize; t++)
		sum += T[lid][t] * total_sum[t];

	x[i_TileSize + lid] = sum;
}
