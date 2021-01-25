#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <vector>
#include <fstream>
#include <opencv2\opencv.hpp>
#define MAX 10
#define PtSum 38
#define MTX_AMT 1000
using namespace std;
using namespace cv;
void Inverse(double *matrix1[], double *matrix2[], int n, double d);
double Determinant(double* matrix[], int n);
double AlCo(double* matrix[], int jie, int row, int column);
double Cofactor(double* matrix[], int jie, int row, int column);
void Empty(double *matrix[], int row, int column);

void lineRegression(vector<Point3f>& points, double& x0, double& y0, double& m, double& n)
{
	double array[MTX_AMT][3], *Matrix[2], *IMatrix[2];
	for (int i = 0; i < points.size(); i++)
	{
		array[i][2] = points[i].z;
		array[i][0] = points[i].x;
		array[i][1] = points[i].y;
	}
	for (int i = 0; i < 2; i++)
	{
		Matrix[i] = new double[2];
		IMatrix[i] = new double[2];
	}
	Empty(Matrix, 2, 2);
	for (int j = 0; j < points.size(); j++)
	{
		*(Matrix[0] + 0) += array[j][2] * array[j][2];
		*(Matrix[0] + 1) += array[j][2];
	}
	*(Matrix[1] + 0) = *(Matrix[0] + 1);
	*(Matrix[1] + 1) = 1.0*points.size();
	double d = Determinant(Matrix, 2);
	Inverse(Matrix, IMatrix, 2, d);
	Empty(Matrix, 2, 2);
	for (int i = 0; i < points.size(); i++)
	{
		*(Matrix[0] + 0) += array[i][0] * array[i][2];
		*(Matrix[0] + 1) += array[i][0];
		*(Matrix[1] + 0) += array[i][1] * array[i][2];
		*(Matrix[1] + 1) += array[i][1];
	}
	m = *(Matrix[0] + 0)**(IMatrix[0] + 0) + *(Matrix[0] + 1)**(IMatrix[1] + 0);
	x0 = *(Matrix[0] + 0)**(IMatrix[0] + 1) + *(Matrix[0] + 1)**(IMatrix[1] + 1);
	n = *(Matrix[1] + 0)**(IMatrix[0] + 0) + *(Matrix[1] + 1)**(IMatrix[1] + 0);
	y0 = *(Matrix[1] + 0)**(IMatrix[0] + 1) + *(Matrix[1] + 1)**(IMatrix[1] + 1);
	for (int i = 0; i < 2; i++)
	{
		delete[] Matrix[i];
		delete[] IMatrix[i];
	}
}

void drawSampleLine(int& x0, int& y0, int& m, int& n, string sampleLine) {
	ofstream fout(sampleLine);
	for (int z = 0; z < 1000; z++) {
		fout << x0 + m*z << " " << y0 + n*z << " " << z << " 255 255 255" << endl;
	}
}


#define pi 3.1415926 

//angle between line and planar
float angleBetweenLineAndPlanarXY(float m, float n) {
	float theta = 0;
	float costheta = 1 / (sqrt(m*m + n*n + 1));
	if (costheta < 0) {
		costheta = -costheta;
	}
	theta = acos(costheta) * 180 / pi;
	theta = 90 - theta;
	return theta;
}

float angleBetweenLines(float m1, float n1, float m2, float n2) {
	float theta = 0;
	float costheta = abs(m1*m2 + n1*n2 + 1) / sqrt((m1*m1 + n1*n1 + 1)*(m2*m2 + n2*n2 + 1));
	theta = acos(costheta) * 180 / pi;
	return theta;
}

void Empty(double *matrix[], int row, int column)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			*(matrix[i] + j) = 0.0;
		}
	}
}

void Inverse(double *matrix1[], double *matrix2[], int n, double d)
{
	int i, j;
	for (i = 0; i<n; i++)
		matrix2[i] = (double *)malloc(n * sizeof(double));
	for (i = 0; i<n; i++)
		for (j = 0; j<n; j++)
			*(matrix2[j] + i) = (AlCo(matrix1, n, i, j) / d);
}

double Determinant(double* matrix[], int n)
{
	double result = 0, temp;
	int i;
	if (n == 1)
		result = (*matrix[0]);
	else
	{
		for (i = 0; i<n; i++)
		{
			temp = AlCo(matrix, n, n - 1, i);
			result += (*(matrix[n - 1] + i))*temp;
		}
	}
	return result;
}

double AlCo(double* matrix[], int jie, int row, int column)
{
	double result;
	if ((row + column) % 2 == 0)
		result = Cofactor(matrix, jie, row, column);
	else result = (-1)*Cofactor(matrix, jie, row, column);
	return result;
}

double Cofactor(double* matrix[], int jie, int row, int column)
{
	double result;
	int i, j;
	double* smallmatr[MAX - 1];
	for (i = 0; i<jie - 1; i++)
		smallmatr[i] = new double[jie - 1];
	for (i = 0; i<row; i++)
		for (j = 0; j<column; j++)
			*(smallmatr[i] + j) = *(matrix[i] + j);
	for (i = row; i<jie - 1; i++)
		for (j = 0; j<column; j++)
			*(smallmatr[i] + j) = *(matrix[i + 1] + j);
	for (i = 0; i<row; i++)
		for (j = column; j<jie - 1; j++)
			*(smallmatr[i] + j) = *(matrix[i] + j + 1);
	for (i = row; i<jie - 1; i++)
		for (j = column; j<jie - 1; j++)
			*(smallmatr[i] + j) = *(matrix[i + 1] + j + 1);
	result = Determinant(smallmatr, jie - 1);
	for (i = 0; i<jie - 1; i++)
		delete[] smallmatr[i];
	return result;
}