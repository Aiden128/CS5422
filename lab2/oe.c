#include <stdio.h>

void printarr(int* arr, int len) {
	for (int i = 0; i < len; i++) {
		printf(" %2d", arr[i]);
	}
	printf("\n");
}

void swap(int* p, int* q) {
	int tmp = *p;
	*p = *q;
	*q = tmp;
}

void oddeven(int* arr, int len) {
	for (int i = len; i > 0; i--) {
		for (int j = i % 2; j < len; j++) {
			if (arr[j] > arr[j+1]) {
				swap(&arr[j], &arr[j+1]);
			}
		}
	}
}

int main() {
	int arr[19] = {32, 0, 33, 39, 52, 73, 26, 50, 93, 30, 46, 76, 93, 16, 82, 18, 13, 51, 70};
	printarr(arr, 19);
	oddeven(arr, 19);
	printarr(arr, 19);
}
