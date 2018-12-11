#ifndef _UTILS_H
#define _UTILS_H

#include <stdio.h>

struct coord {
    int i, j;
};

typedef struct coord *coord;

struct _p_COO {
    int m, n, NZ;
    struct coord *coords;
    double *data;
};

typedef struct _p_COO *COO;

struct _p_CSR {
    int m, n, NZ;
    int *I;
    int *J;
    double *data;
};

typedef struct _p_CSR *CSR;

void alloc_sparse(int, int, int, COO*);
void free_sparse(COO*);
void alloc_sparse_csr(int, int, int, CSR*);
void free_sparse_csr(CSR*);
void alloc_dense(int, int, double **);
void free_dense(double **);
void zero_dense(int, int, double *);

void convert_sparse_to_dense(const COO, double **);
void convert_dense_to_sparse(const double *, int, int, COO *);

void convert_coo_to_csr(const COO, CSR *);
void convert_csr_to_coo(const CSR, COO *);

void read_sparse(const char *, COO *);
void write_sparse(FILE *, COO);
void write_sparse_csr(FILE *, CSR);
void read_sparse_binary(const char *, COO *);
void write_sparse_binary(FILE *, COO);
void print_sparse(COO);
void print_sparse_csr(CSR);
void random_matrix(int, int, double, COO *);


#endif
