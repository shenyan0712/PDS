///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//同时使用local memory和向量优化


/*
step 1. 计算T_cdj=A_cdj-L_abj*Lt_aj.
*/
__kernel void chol_v4_step1(
	const int j,
	const int SR,		//start pos of LA,LB in A 
	const int SC,		//start column of TC in A
	const int MN,
	__global VEC *A,
	__global VEC *DBIs,
	__global double *ret
)
{
	__local VEC subLAB[TileSize_R][TileSize_C];		//[tileSize, tileSize/VEC_WIDTH]
	__local VEC subLA[TileSize_R][TileSize_C];		//[tileSize, tileSize/VEC_WIDTH]
	VEC sum = 0;

	const int u = get_local_id(0);
	const int v = get_local_id(1);
	const int offset_LA = SR*MN;
	const int offset_LAB = offset_LA + get_group_id(0)*TileSize_R*MN;
	const int addr1 = u*MN + v;
	const int vVEC = v*VEC_WIDTH;

	for (int n = 0; n < j; n++) {
		//load subLAB, subLA.
		subLAB[u][v] = A[offset_LAB + addr1 + n*TileSize_C];
		subLA[u][v] = A[offset_LA + addr1 + n*TileSize_C];
		barrier(CLK_LOCAL_MEM_FENCE);	//等待块加载完

		//subLAB的u行和subLA的v行的点积
		for (int k = 0; k < TileSize_C; k++)
		{
#if VEC_WIDTH==1
			//sum += subLAB[u][k] * subLA[vVEC][k];
			sum = fma(subLAB[u][k], subLA[vVEC][k], sum);  //fma乘加比上面的运算更快
#elif VEC_WIDTH==2
			sum.x += dot(subLAB[u][k], subLA[vVEC][k]);
			sum.y += dot(subLAB[u][k], subLA[vVEC + 1][k]);
#elif VEC_WIDTH==4
			sum.x += dot(subLAB[u][k], subLA[v*VEC_WIDTH][k]);
			sum.y += dot(subLAB[u][k], subLA[v*VEC_WIDTH + 1][k]);
			sum.z += dot(subLAB[u][k], subLA[v*VEC_WIDTH + 2][k]);
			sum.w += dot(subLAB[u][k], subLA[v*VEC_WIDTH + 3][k]);
#endif
		}
		barrier(CLK_LOCAL_MEM_FENCE);	//等待块加载完
	}
	A[offset_LAB + addr1 + SC] -= sum; //subLAB[u][v];
}

/*
计算第j列对角块的L_jj, inv(L_jj）
*/
__kernel void chol_v4_step2(
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

	DB[uvAddr] = A[xyAddr];
	barrier(CLK_LOCAL_MEM_FENCE);

	flag = compute_Ljj(DB, uvAddr, tileSize, u, v);
	compute_LjjInv(DB, DBI, uvAddr, tileSize, u, v);

	A[xyAddr] = DB[uvAddr];
	DBIs[S*tileSize + uvAddr] = DBI[uvAddr];
}


/*
使用inv(L_jj)计算剩余L_dj=T_dj*inv(L_jj)
*/
__kernel void chol_v4_step3(
	const int j,			//当前处理的列
	const int tileSize,
	const int matSize,
	const int TN,
	const int MN,
	__global VEC *A,
	__global VEC *DBIs,
	__global double *ret
)
{
	__local VEC T[TileSize*TileSize];
	__local VEC DBI[TileSize*TileSize];

	int tx = get_group_id(0) + j + 1;
	int u = get_local_id(0);
	int v = get_local_id(1);
	int uTN = u*TN;
	int uTNv = uTN + v;
	int vtTN = (v*VEC_WIDTH)*TN;
	int addr1 = tx*tileSize*MN + u*MN + j*TN + v;

	//copy A[]-->T,  DBIs[] -->DBI
	T[uTNv] = A[addr1];
	DBI[uTNv] = DBIs[j*tileSize*TN + uTNv];
	barrier(CLK_LOCAL_MEM_FENCE);

	VEC sum = 0;
	for (int k = 0; k < TN; k++) {
		int uTNk = uTN + k;
		int taddr = vtTN + k;
#if VEC_WIDTH==1
		//sum += T[uTNk] * DBI[taddr];
		sum = fma(T[uTNk], DBI[taddr], sum);
#elif VEC_WIDTH==2
		sum.x += dot(T[uTNk], DBI[taddr]); taddr += TN;
		sum.y += dot(T[uTNk], DBI[taddr]); taddr += TN;
#elif VEC_WIDTH==4
		sum.x += dot(T[uTNk], DBI[taddr]); taddr += TN;
		sum.y += dot(T[uTNk], DBI[taddr]); taddr += TN;
		sum.z += dot(T[uTNk], DBI[taddr]); taddr += TN;
		sum.w += dot(T[uTNk], DBI[taddr]);
#endif
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	A[addr1] = sum;
}