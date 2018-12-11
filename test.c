#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "utils.h"

int main(int argc, char **argv)
{
    // COO A, AT, B;
    COO A, C, B, AB;
    // CSR csr, csrt;
    int m, k;
    int pass = 0;
    int x[10], y[10];

    #pragma acc parallel loop 
    for (int j = 0; j < 10; j++) { 
        x[j] = j; 
        y[j] = -j; 
    }

    printf("%d", x[9]);

    m = 5;
    k = 5;

    random_matrix(m, k, 0.1, &A);
    csr_mm_multiply(A, A, &C);
    coo_mm_multiply_acc(A, A, &AB);
    coo_mm_multiply(A, A, &B);
    // // random_matrix(m, k, 0.3, &B);

    // // transpose_coo_acc(B, &AT);

    // convert_coo_to_csr(A, &csr);
    // csr_transpose(csr, &csrt);
    // convert_csr_to_coo(csrt, &AT);

    print_sparse(A);
    print_sparse(C);
    print_sparse(B);
    print_sparse(AB);
    // print_sparse_csr(csr);
    // print_sparse_csr(csrt);
    // print_sparse(AT);
    // // print_sparse(B);

    free_sparse(&A);
     free_sparse(&B);
     free_sparse(&C);
     free_sparse(&AB);
    // free_sparse_csr(&csr);
    // free_sparse_csr(&csrt);

    return pass;
}