#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "utils.h"

int main(int argc, char **argv)
{
    COO A, AT, B;
    int m, k;
    int pass = 0;

    m = 5;
    k = 5;

    random_matrix(m, k, 0.1, &A);
    random_matrix(m, k, 0.3, &B);
    transpose_coo(B, &AT);

    print_sparse(A);
    print_sparse(AT);
    print_sparse(B);

    free_sparse(&A);
    free_sparse(&B);
    free_sparse(&AT);

    return pass;
}