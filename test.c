#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "utils.h"

int main(int argc, char **argv)
{
    COO A, AT;
    int m, k;
    int pass = 0;

    m = 20;
    k = 50;

    random_matrix(m, k, 0.1, &A);
    transpose_coo(A, &AT);

    print_sparse(A);
    print_sparse(AT);

    free_sparse(&A);
    free_sparse(&AT);

    return pass;
}