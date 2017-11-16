///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/*
计算tileSize*tileSize对角块的cholesky分解
全部操作都在局部块DB内
*/
inline int compute_Ljj(local double* DB, int uvAddr, int tileSize, int u, int v)
{ 
	//int addr = u*tileSize + v;
	int addr2;
	int addr3;
	int ret = 0;
	double sum=0;
	for (int j = 0; j< tileSize; j++)	//单列的处理
	{
		//对角元, d_v=a[v,k]*a[v,k] k=1,...,v-1
		sum = DB[uvAddr];
		if(u==v && j==v)
		{
			if (v > 0) {
				addr2 = v*tileSize;
				for (int n = 0;n < v; n++) {
					//sum -= DB[addr2] * DB[addr2];
					sum = -fma(DB[addr2], DB[addr2], -sum);
					addr2++;
				}
			}
			if (sum<0) ret = -1;
			else DB[uvAddr] = sqrt(sum);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		//非对角元, t_v=a[u,k]*a[v,k] k=1,...,v-1
		if (u > v && j == v)
		{
			if (v > 0) {
				addr2 = u*tileSize;
				addr3 = v*tileSize;
				for (int n = 0;n < v; n++) {
					sum -= DB[addr2++] * DB[addr3++];
				}
			}
			DB[uvAddr] = sum / DB[v*tileSize + v];
		}
		if(u<v && j==v)
			DB[uvAddr] = 0;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	return ret;
}

/*
计算tileSize*tileSize对角块Ljj的逆
*/
inline void compute_LjjInv(local double* DB,local double* DBI, int uvAddr, int tileSize, int u, int v)
{ 
	//int addr = u*tileSize + v;
	//int addr2;
	//int addr3;
	double sum = 0;
	for (int k = 0; k < tileSize; k++)	//单条diagonal的处理, (main diagonal, non-main diagonal)
	{
		if ((u - v) == k) {		//确定是该对角线上的元素
			if (k == 0)
				DBI[uvAddr] = 1 / DB[uvAddr];
			else{ 
				for (int s = 0; s < u; s++)	//sum(L_is*X_sj )
					sum -= DB[u*tileSize + s] * DBI[s*tileSize + v];
				DBI[uvAddr] = sum / DB[u*tileSize + u];
			}
		}
		if (u < v) DBI[uvAddr] = 0;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

}