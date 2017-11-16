
#include "misc.h"


#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define MAX_LINE_DATA_NUM	1000*100
#define MAX_LINE_CHARS		1000*100*20


char line_buff[MAX_LINE_CHARS];

int readDataFromLine(FILE *fp, double *data, int *nDatas);

/*
read m*n dense matrix from txt file
*/
int read_dmat(char *fileName, int m,int n, double* mat)
{
    int ret;
    int nDatas;
    FILE *fp;

    //double *datas = new double[MAX_LINE_DATA_NUM];
    //读取矩阵文件
#ifdef _MSC_VER
    if (fopen_s(&fp, fileName, "r") != 0)
    {
#else
    if ((fp=fopen(fileName, "r")) != 0)
    {
#endif // _MSC_VER
        fprintf(stderr, "cannot open file %s, exiting\n", fileName);
        system("pause");
        exit(1);
    }

    //从每一行读取数据
    int cnt = 0;
    while (1)
    {
        ret = readDataFromLine(fp, &mat[cnt*n], &nDatas);
        if (ret != 1) cnt++;
        else break; //文件尾
        if (n != nDatas) return -1;
    }
    if (cnt != m) return -1;

    //delete[] datas;
    return 0;
}

/*
从文件的当前行读取nDatas个数据, 数据以'\t' ','或空格分割
======输入==========
fp		==>文件描述符
data	==>提供的MAX_LINE_DATA_NUM个用于存放数据的空间
======输出==========

======返回值========
==1 该行为文件尾， ==0 表示正常行 ==2
*/
int readDataFromLine(FILE *fp, double *data, int *nDatas)
{
    char *ret;
    int nChars, pos, start;
    double val;

    *nDatas = 0;
    start = 0;
    pos = 0;
    //将该行读到line_buff
    ret = fgets(line_buff, MAX_LINE_CHARS - 1, fp);
    if (ret == NULL) return 1;
    nChars = (int)strlen(line_buff); //包含'\n'
    while (pos < nChars)
    {
        //跳过多余的分割符
        while ((line_buff[pos] == '\t' || line_buff[pos] == ',' || line_buff[pos] == ' ') && line_buff[pos] != '\n')
            pos++;
        if (line_buff[pos] == '\n' || line_buff[pos] == '#') break;

        //pos开始的字符可能是一个数字, 进行解析
#ifdef _MSC_VER
        sscanf_s(&line_buff[pos], "%lf", &val);
#else
        sscanf(&line_buff[pos], "%lf", &val);
#endif // _MSC_VER
        data[(*nDatas)++] = val;

        //跳过数据，直到遇到分隔符
        while (line_buff[pos] != '\t' && line_buff[pos] != ',' && line_buff[pos] != ' ' && line_buff[pos] != '\n')
            pos++;
    }

    return 0;
}
