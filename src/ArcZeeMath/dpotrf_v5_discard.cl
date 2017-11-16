///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//使用local memory, 适用于不被块尺寸整除的矩阵


/*
step 1. 计算T_cdj=A_cdj-L_abj*Lt_aj.
*/
__kernel void chol_v5_step1(
	const int j,
	const int SR,		//start row of LA,LB in A 
	const int SC,		//start column of TC in A
	const int matSize,
	__global double *A,
	__global double *DBIs,
	__global double *ret
)
{
	__local double subLAB[TileSize][TileSize];		//[tileSize, tileSize/VEC_WIDTH]
	__local double subLA[TileSize][TileSize];		//[tileSize, tileSize/VEC_WIDTH]
	double sum = 0;
	const int u = get_local_id(0);
	const int v = get_local_id(1);
	const int x = (get_group_id(0) + j)*TileSize + u;
	const int y = (get_group_id(1) + j)*TileSize + v;
	const int offset_LA = SR*matSize;
	const int offset_LAB = offset_LA + get_group_id(0)*TileSize*matSize;
	const int addr1 = u*matSize + v;

	for (int n = 0; n < j; n++) {
		//load subLAB, subLA.
		subLAB[u][v] = A[offset_LAB + addr1 + n*TileSize];
		subLA[u][v] = A[offset_LA + addr1 + n*TileSize];
		barrier(CLK_LOCAL_MEM_FENCE);	//等待块加载完

		//subLAB的u行和subLA的v行的点积
		for (int k = 0; k < TileSize; k++)
		{
			//sum += subLAB[u][k] * subLA[v][k];
			sum = fma(subLAB[u][k], subLA[v][k], sum);  //fma乘加比上面的运算更快
		}
		barrier(CLK_LOCAL_MEM_FENCE);	//等待块加载完
	}
	if( y<matSize)
		A[offset_LAB + addr1 + SC] -= sum; //subLAB[u][v];
}

/*
计算第j列对角块的L_jj, inv(L_jj）
*/
__kernel void chol_v5_step2(
	const int j,			//当前处理的列
	const int tileSize,
	const int matSize,
	__global double *A,
	__global double *DBIs,
	__global double *ret
)
{
	__local double DB[TileSize*TileSize];		//tileSize*tileSize, 用于存储
	__local double DBI[TileSize*TileSize];		//tileSize*tileSize

	int flag;
	int S = j*tileSize;
	int u = get_local_id(0);
	int v = get_local_id(1);

	int x = u + S;
	int y = v + S;
	int uvAddr = u*tileSize + v;
	int xyAddr = x*matSize + y;

	DB[uvAddr] = 0.0;
	if (x < matSize && y < matSize)
		DB[uvAddr] = A[xyAddr];
	barrier(CLK_LOCAL_MEM_FENCE);

	flag = compute_Ljj(DB, uvAddr, tileSize, u, v);
	compute_LjjInv(DB, DBI, uvAddr, tileSize, u, v);

	if(x < matSize && y < matSize)
		A[xyAddr] = DB[uvAddr];
	DBIs[S*tileSize + uvAddr] = DBI[uvAddr];
}


/*
使用inv(L_jj)计算剩余L_dj=T_dj*inv(L_jj)
*/
__kernel void chol_v5_step3(
	const int j,			//当前处理的列
	const int tileSize,
	const int matSize,		//MN
	__global double *A,
	__global double *DBIs,
	__global double *ret
)
{
	__local double T[TileSize*TileSize];
	__local double DBI[TileSize*TileSize];

	int tx = get_group_id(0) + j + 1;
	int u = get_local_id(0);
	int v = get_local_id(1);
	int x = tx*tileSize + u;
	int utileSize = u*tileSize;
	int utileSize_v = utileSize + v;
	int vtileSize = v*tileSize;
	int addr1 = x*matSize + j*tileSize + v;

	//copy A[]-->T,  DBIs[] -->DBI
	T[utileSize_v] = A[addr1];
	DBI[utileSize_v] = DBIs[j*tileSize*tileSize + utileSize_v];
	barrier(CLK_LOCAL_MEM_FENCE);

	double sum = 0;
	for (int k = 0; k < tileSize; k++) {
		//sum += T[uTNk] * DBI[taddr];
		sum = fma(T[utileSize + k], DBI[vtileSize + k], sum);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (x < matSize)
		A[addr1] = sum;
}