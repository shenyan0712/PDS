
#include<iostream>
#include<mkl.h>

#include<arczee.h>
#include<spdsys.h>
#include<cldef.h>
#include<cldev.h>
#include<misc.h>

void dmake_hpd(int N, double* A, int lda);
double compare2D(int N, double*A, double*B);
double compare1D(int N, double*x, double*y);
void print2D(int M, int N, double*A);
void print1D(int N, double*x);
void printMatPart(int matSize, int tileSize, double*A, int bx, int by);
void printDBs(int N, int tileSize, double *DBs);
void gen_mat12(double *A);
void gen_mat16(double *A);

using namespace std;
#define USE_SMALL_MAT	0

#if USE_SMALL_MAT==1
int N = 16;
#else
int N = 5130;
#endif

int main()
{
	cout << "hello world" << endl;

	//==========openCL initialization
	cldev cd;
	cd.init(true);
	//selecte devices
	if (cd.selectPfWithMostDev(CL_DEVICE_TYPE_GPU, 1.2, 1)) {
		cout << "No devices satisfy the requirement.";
	}
	arczee_init(cd);

	//generate SPD matrix
	int lda = N;
	int iseed[4] = { 0,0,0, 1 };
	double *A = new double[N * N];
	double *A_mkl = new double[N*N];
	double *DBIs = new double[N * TileSize];
	double *b = new double[N];
	double *b_bak = new double[N];

	int cols = (int)ceil((float)N / TileSize);
	int vMatSize = cols*TileSize;
	clBufferEx<double> A_buf = clBufferEx<double>(cd.get_context(), cd.get_queue(0), N *(N + TileSize), MODE_NO_SVM);
	clBufferEx<double> DBIs_buf = clBufferEx<double>(cd.get_context(), cd.get_queue(0), vMatSize* TileSize, MODE_NO_SVM);
	clBufferEx<double> b_buf = clBufferEx<double>(cd.get_context(), cd.get_queue(0), N, MODE_NO_SVM);
	clBufferEx<double> x_buf = clBufferEx<double>(cd.get_context(), cd.get_queue(0), N, MODE_NO_SVM);
	double s_initial, s_elapsed;
	double gflops;

	//***** generate Matrix
	if (USE_SMALL_MAT) {
		gen_mat16(A);
		//print2D(N, N, A);
		for (int i = 0; i < N; i++)
			b[i] = i + 1;
		memcpy(b_bak, b, N * sizeof(double));

		//set elem of matrix to be (0,1) with normal distribution.
		//LAPACKE_dlarnv(1, iseed, N*N, A);
		//make A symmetric definite
		//dmake_hpd(N, A, lda);

		print2D(N, N, A);
	}
	else {
		//set elem of matrix to be (0,1) with normal distribution.
		LAPACKE_dlarnv(1, iseed, N*N, A);
		//make A symmetric definite
		dmake_hpd(N, A, lda);
		//****** generate b randomly
		LAPACKE_dlarnv(1, iseed, N, b);
		memcpy(b_bak, b, N * sizeof(double));
	}

	dlacpy("b", &N, &N, A, &lda, A_mkl, &lda);	//not 'U' and 'L' mean all the matrix is copied.


	//###############################################################################################
	// Cholesky test
	//###############################################################################################
	//gflops for Cholesky;
	gflops = (FMULS_POTRF(N) + FADDS_POTRF(N)) / 1e9;	// pow(N, 3) / (3.0 * 1e9);	//get GFlops of the decomposition for N*N matrix

	s_elapsed = 0;
	for (int i = 0; i < 1; i++) {
		if (i % 10 == 0)
			printf("iter: %d\n", i);
		A_buf.write(0, (double*)A, N*N);
		s_initial = dsecnd();
		//dpotrf_v1_cl(A_buf, N, TileSize, cd, 0, DBIs_buf);		//原始版本
		//dpotrf_v2_cl(A_buf, N, TileSize, cd, 0, DBIs_buf);		//使用local memory优化
		//dpotrf_v3_cl(A_buf, N, TileSize, cd, 0, DBIs_buf);		//使用向量优化
		dpotrf_v4_cl(A_buf, N, TileSize, cd, 0, DBIs_buf);		//同时使用local memory和向量优化

		s_elapsed += (dsecnd() - s_initial);
	}
	A_buf.read(0, (double*)A, N*N);
	DBIs_buf.read(0, (double*)DBIs, N*TileSize);

	if (USE_SMALL_MAT) {
		print2D(N, N, A);
		printDBs(N, TileSize, DBIs);
	}

	double gpu_perf = 10 * gflops / (s_elapsed);	//get the performance, GFlop/s
	printf("perf: %.3f GLOPS/s     time cost:%.3f\n", gpu_perf, s_elapsed);

	//compare
	int info;
	dpotrf("U", &N, A_mkl, &lda, &info);
	double diff = compare2D(N, A, A_mkl);
	printf("err:%e\n", diff);
	if (diff > 1e-3 || info!=0) {
		printf("The implementation must be wrong.\n");
		system("pause");
		return 0;
	}

	system("pause");
	return 0;
	//###############################################################################################
	// Triangular Equation test
	//###############################################################################################
	//gflops for solving triangular system
	gflops = (FMULS_POTRS(N, 1) + FADDS_POTRS(N, 1)) / 1e9;	// pow(N, 3) / (3.0 * 1e9);	//get GFlops of the decomposition for N*N matrix

	s_elapsed = 0;
	for (int i = 0; i < 1; i++) {
		b_buf.write(0, (double*)b_bak, N);
		s_initial = dsecnd();

		//dpotrs_v1(A_buf, x_buf, b_buf, DBIs_buf, N, TileSize, cd, 0);		//采用reduction, 未采用VEC
		//dpotrs_v2(A_buf, x_buf, b_buf, DBIs_buf, N, TileSize, cd, 0);
		dpotrs_v3(A_buf, b_buf, DBIs_buf, N, TileSize, cd, 0);

		s_elapsed += (dsecnd() - s_initial);
	}

    //A_buf.read(0, (double*)A, N*N);
    b_buf.read(0, (double*)b, N);
	if (USE_SMALL_MAT) {
		print1D(N, b);
	}

	gpu_perf = 100 * gflops / (s_elapsed);	//get the performance, GFlop/s
	printf("perf[trig sys]: %.3f GLOPS/s     time cost:%.3f\n", gpu_perf, s_elapsed);

	//compare the result with MKL
	int rhs = 1;
	dpotrs("U", &N, &rhs, A_mkl, &lda, b_bak, &N, &info);
	//print1D(N, b);
	diff = compare1D(N, b, b_bak);
	printf("err:%e\n", diff);
	if (diff > 1e-3) {
		printf("The implementation must be wrong.\n");
		system("pause");
		return 0;
	}

    system("pause");
    return 0;
}



void dmake_hpd(int N, double* A, int lda)
{
	int i, j;
	for (i = 0; i < N; ++i) {
		A[i*N + i] = A[i*N + i]*N+N;  //+N
		for (j = 0; j < i; ++j) {
			A[j*N + i] = A[i*N + j];
		}
	}
}

double compare2D(int N, double*A, double*B)
{
	int i, j;
	double sum = 0;
	double diff;
	for (i = 0; i < N; i++) {
		for (j = 0; j <= i; j++) {
			diff = A[i*N + j] - B[i*N + j];
			sum += diff*diff;
		}
	}
	return sqrt(sum);
}

double compare1D(int N, double*x, double*y)
{
	double sum = 0; 
	double diff;
	for (int i = 0; i < N; i++) {
		diff = abs(x[i] - y[i]);
		sum += diff*diff;
	}
	return sqrt(sum);
}

void print1D(int N, double*x)
{
	printf("x:\n");
	for (int i = 0; i < N; i++)	{
		printf("%.3f  ", x[i]);
	}
	printf("\n");
}

void print2D(int M, int N, double*A)
{
	printf("A:\n");
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
			printf("%.4f  ", A[i*M + j]);
		printf("\n");
	}
	printf("\n");
}

void printMatPart(int matSize, int tileSize, double*A, int bx, int by)
{
	printf("A part():\n", bx,by);
	int xbase = bx*tileSize*matSize;
	int ybase = by*tileSize;
	for (int i = 0; i < tileSize; i++)
	{
		for (int j = 0; j < tileSize; j++)
			printf("%.3f  ", A[xbase+i*matSize +ybase+ j] );
		printf("\n");
	}
	printf("\n");
}


void printDBs(int N,int tileSize, double *DBs) 
{
	int cols = (int)ceil((float)N / tileSize);

	printf("DBIs:\n");
	for (int i = 0; i < cols; i++)
	{
		for (int u = 0; u < tileSize; u++)
		{
			for (int v = 0; v < tileSize; v++)
				printf("%.3f  ", DBs[i*tileSize*tileSize + u*tileSize + v]);
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}


void gen_mat12(double *A)
{
	double B[12 * 12] = {
		25,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,
		2,	25,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,
		3,	4,	25,	6,	7,	8,	9,	10,	11,	12,	13,	14,
		4,	5,	6,	25,	8,	9,	10,	11,	12,	13,	14,	15,
		5,	6,	7,	8,	25,	10,	11,	12,	13,	14,	15,	16,
		6,	7,	8,	9,	10,	25,	12,	13,	14,	15,	16,	17,
		7,	8,	9,	10,	11,	12,	25,	14,	15,	16,	17,	18,
		8,	9,	10,	11,	12,	13,	14,	25,	16,	17,	18,	19,
		9,	10,	11,	12,	13,	14,	15,	16,	25,	18,	19,	20,
		10,	11,	12,	13,	14,	15,	16,	17,	18,	25,	20,	21,
		11,	12,	13,	14,	15,	16,	17,	18,	19,	20,	25,	22,
		12,	13,	14,	15,	16,	17,	18,	19,	20,	21,	22,	25,
	};
	for (int i = 0; i < 12 * 12; i++)
		A[i] = B[i];
}

void gen_mat16(double *A)
{
	double B[16 * 16] = {
		1.2330,    1.0988 ,   2.0308 ,   1.5374,    0.0697,    0.4985,    0.4034,    1.1333,    3.4123,    0.6954,    0.3183 ,   0.2191,    0.4504 ,   1.5761 ,   0.8100,    1.2740,
		1.0988,    2.6209 ,   4.6327,    2.1020,    0.3363 ,   1.6517 ,   0.4796,    2.4479 ,   3.4332,    2.1218 ,   1.5150 ,   1.0329,    1.9766,    1.7517,    1.8749,    1.5014,
		2.0308,    4.6327 ,   8.4526 ,   4.4116 ,   0.8935 ,   2.9271 ,   1.6095,    5.1579 ,   7.2839,    5.0401 ,   3.1312 ,   1.9307,    3.9410,    3.7700,    3.5926,    3.5158,
		1.5374,    2.1020,    4.4116 ,   7.7322,    2.6375 ,   2.0587,    6.1747,    4.8379 ,   7.6677,    7.0255 ,   3.3898 ,   3.2917,    3.2474,    6.1269,    3.3819,    4.5511,
		0.0697,    0.3363 ,   0.8935 ,   2.6375 ,   2.2049 ,   1.5034 ,   2.7229,    1.8240 ,   2.6993,    3.0285 ,   2.6559 ,   1.5527,    1.4255,    4.2244,    1.8750,    2.9000,
		0.4985,    1.6517 ,   2.9271 ,   2.0587 ,   1.5034,    2.1022,    1.5383,    2.0952 ,   2.9409,    2.4176 ,   2.9104 ,   1.5928,    1.7656,    3.6304,    2.3400,    2.6622,
		0.4034,    0.4796 ,   1.6095 ,   6.1747 ,   2.7229,    1.5383,    7.5266,    5.6125 ,   5.6996,    7.5079 ,   5.1341 ,   4.0305,    2.9171,    6.7921,    3.0237,    5.2809,
		1.1333,    2.4479 ,	  5.1579 ,   4.8379 ,   1.8240 ,   2.0952 ,   5.6125 ,   9.5240  , 10.3434 ,   9.7117  ,  5.9541  ,  3.9022 ,   5.0775 ,   7.8218 ,   4.2380 ,   6.3923,
		3.4123,    3.4332,    7.2839 ,   7.6677,    2.6993,    2.9409 ,   5.6996,   10.3434 ,  21.9758,   13.2542 ,   6.4318 ,   3.2473,    6.0882,   16.0268,    6.8271,   12.0553,
		0.6954,    2.1218,    5.0401,    7.0255,    3.0285,    2.4176 ,  7.5079 ,   9.7117  , 13.2542 ,  13.9111  ,  7.3688  ,  4.5495 ,   6.4860 ,  12.0901 ,   6.1054 ,   9.1223,
		0.3183,    1.5150 ,   3.1312 ,   3.3898,    2.6559,    2.9104 ,  5.1341 ,   5.9541  ,  6.4318 ,   7.3688  ,  8.5694  ,  4.5641 ,   4.0128 ,  10.8223 ,   5.9122 ,   8.6050,
		0.2191,    1.0329 ,   1.9307 ,   3.2917 ,   1.5527 ,   1.5928 ,   4.0305,    3.9022 ,   3.2473,    4.5495 ,   4.5641 ,   3.9034,    2.9988,    7.0015,    4.1083,    4.1505,
		0.4504 ,   1.9766 ,   3.9410 ,   3.2474 ,   1.4255,    1.7656 ,   2.9171,    5.0775 ,   6.0882,    6.4860 ,   4.0128 ,   2.9988,    8.1715,    8.8871,    6.2347,    6.5274,
		1.5761 ,   1.7517 ,   3.7700 ,   6.1269,    4.2244,    3.6304 ,   6.7921,    7.8218 ,  16.0268,   12.0901 ,  10.8223 ,   7.0015,    8.8871,   35.6283,   12.9400,   16.3629,
		0.8100 ,   1.8749 ,   3.5926 ,   3.3819,    1.8750,    2.3400 ,   3.0237,    4.2380 ,   6.8271,    6.1054 ,   5.9122 ,   4.1083,    6.2347,   12.9400,    9.6056,    7.2033,
		1.2740 ,   1.5014 ,   3.5158 ,   4.5511,    2.9000 ,   2.6622 ,   5.2809,    6.3923 ,  12.0553,    9.1223 ,   8.6050 ,   4.1505,    6.5274,   16.3629,    7.2033,   19.2204
	};
	for (int i = 0; i < 16 * 16; i++)
		A[i] = B[i];
}