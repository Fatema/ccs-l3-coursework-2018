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
    int m, k, j;
    int pass = 0;
    int x[10], y[10];

    #pragma acc parallel loop 
    for (j = 0; j < 10; j++) {
        x[j] = j; 
        y[j] = -j; 
    }

    m = 50;
    k = 30;

    random_matrix(m, k, 0.1, &A);
//    print_sparse(A);

//    transpose_coo_acc(A, &B);
//    print_sparse(B);

    optimised_sparsemm_sum(A, A, A , A, A, A, &C);
//    coo_mm_multiply_acc(A, A, &AB);
    coo_mm_multiply(A, A, &B);
    // // random_matrix(m, k, 0.3, &B);


    // convert_coo_to_csr(A, &csr);
    // csr_transpose(csr, &csrt);
    // convert_csr_to_coo(csrt, &AT);

    print_sparse(C);
    print_sparse(B);
//    print_sparse(AB);
    // print_sparse_csr(csr);
    // print_sparse_csr(csrt);
    // print_sparse(AT);
    // // print_sparse(B);

    free_sparse(&A);
     free_sparse(&B);
     free_sparse(&C);
//     free_sparse(&AB);
    // free_sparse_csr(&csr);
    // free_sparse_csr(&csrt);

    return pass;
}