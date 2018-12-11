#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "utils.h"

int main(int argc, char **argv)
{
    // COO A, AT, B;
    COO A;
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

    random_matrix(m, k, 0.3, &A);
    // // random_matrix(m, k, 0.3, &B);

    // // transpose_coo(B, &AT);

    // convert_coo_to_csr(A, &csr);
    // transpose_csr(csr, &csrt);
    // convert_csr_to_coo(csrt, &AT);

    print_sparse(A);
    // print_sparse_csr(csr);
    // print_sparse_csr(csrt);
    // print_sparse(AT);
    // // print_sparse(B);

    free_sparse(&A);
    // free_sparse(&B);
    // free_sparse(&AT);
    // free_sparse_csr(&csr);
    // free_sparse_csr(&csrt);

    return pass;
}