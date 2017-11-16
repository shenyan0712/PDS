///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//使用向量优化


/*
step 1. 计算T_cdj=A_cdj-L_abj*Lt_aj.
*/
__kernel void chol_v3_step1(
	const int j,			//当前处理的列
	const int tileSize,
	const int matSize,
	__global VEC *A,
	__global VEC *DBIs,
	__global double *ret
)
{
	int npart = matSize / VEC_WIDTH;
	int S = j*tileSize;				//
	int N = S / VEC_WIDTH;
	int x = get_global_id(0) + S;	//第x行
	int v = get_global_id(1);
	int by = get_global_id(1) + N;	//

	int addr1 = x*npart;
	int addr2 = S + v*VEC_WIDTH;

	//第u行的[0,j*tileSize)和第v行的[0,j*tileSize)的点积
	VEC sum = 0;
	for (int k = 0; k < N; k++)
	{
#if VEC_WIDTH==2
		sum.x += dot(A[addr1 + k] , A[(addr2)*npart + k]);
		sum.y += dot(A[addr1 + k], A[(addr2+1)*npart + k]);
#elif VEC_WIDTH==4
		sum.x += dot(A[addr1 + k], A[(addr2)*npart + k]);
		sum.y += dot(A[addr1 + k], A[(addr2 + 1)*npart + k]);
		sum.z += dot(A[addr1 + k] ,A[(addr2+2)*npart + k]);
		sum.w += dot(A[addr1 + k] , A[(addr2 + 3)*npart + k]);
#endif
	}
	A[x*npart + by] -= sum;
	//A[x*npart + by] =(double4)(1.11,2.22,3.33,4.44);
}



/*
计算第j列对角块的L_jj, inv(L_jj）
*/
__kernel void chol_v3_step2(
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
}

/*
使用inv(L_jj)计算剩余L_dj=T_dj*inv(L_jj)
*/
__kernel void chol_v3_step3(
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
	/*
	int TN = tileSize / VEC_WIDTH;
	int MN = matSize / VEC_WIDTH;	//
	int tx = get_group_id(0) + j + 1;
	int u = get_local_id(0);
	int v = get_local_id(1);
	int uTN = u*TN;
	int vtTN = (v*VEC_WIDTH)*TN;
	int addr1 = tx*tileSize*MN + u*MN + j*TN + v;

	//copy A[]-->T,  DBIs[] -->DBI
	T[uTN + v] = A[addr1];
	DBI[uTN + v] = DBIs[j*tileSize*TN + uTN + v];
	barrier(CLK_LOCAL_MEM_FENCE);

	VEC sum = 0;
	for (int k = 0; k < TN; k++) {
		int uTNk = uTN + k;
		int vtTNk = vtTN + k;
		sum.x += dot(T[uTNk], DBI[vtTNk + 0 * TN]);
		sum.y += dot(T[uTNk], DBI[vtTNk + 1 * TN]);
#if VEC_WIDTH==4
		sum.z += dot(T[uTNk], DBI[vtTNk + 2 * TN]);
		sum.w += dot(T[uTNk], DBI[vtTNk + 3 * TN]);
#endif
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	A[addr1] = sum;
	*/

	///*
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
	A[x*matSize + y] =  sum; // DBIs[vAddr + 1];
	//*/
}