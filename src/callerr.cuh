#include<stdio.h>
void callErr(const char* str)
{
	printf("Reset gpu");
	cudaDeviceReset();
}
