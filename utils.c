#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

/*
 * Allocate a sparse matrix in coordinate format.
 * m - number of rows
 * n - number of columns
 * NZ - number of nonzeros
 * sparse - newly allocated matrix.
 */
void alloc_sparse(int m, int n, int NZ, COO *sparse)
{
    COO sp = calloc(1, sizeof(struct _p_COO));
    sp->m = m;
    sp->n = n;
    sp->NZ = NZ;
    sp->coords = calloc(NZ, sizeof(struct coord));
    sp->data = calloc(NZ, sizeof(double));
    *sparse = sp;
}

/*
 * Free a sparse matrix.
 * sparse - sparse matrix, may be NULL
 */
void free_sparse(COO *sparse)
{
    COO sp = *sparse;
    if (!sp) {
        return;
    }
    free(sp->coords);
    free(sp->data);
    free(sp);
    *sparse = NULL;
}

/*
 * Convert a sparse matrix to dense format in column major format.
 *
 * sparse - The sparse matrix to convert
 * dense - pointer to output dense matrix (will be allocated)
 */
void convert_sparse_to_dense(const COO sparse, double **dense)
{
    int n;
    *dense = calloc(sparse->m * sparse->n, sizeof(double));
    for (n = 0; n < sparse->NZ; n++) {
        int i, j;
        i = sparse->coords[n].i;
        j = sparse->coords[n].j;
        (*dense)[j * sparse->m + i] = sparse->data[n];
    }
}

/*
 * Convert a dense matrix in column major format to sparse.
 * Entries with absolute value < 1e-15 are flushed to zero and not
 * stored in the sparse format.
 *
 * dense - the dense array
 * m - number of rows
 * n - number of columns
 * sparse - output sparse matrix (allocated by this routine)
 */
void convert_dense_to_sparse(const double *dense, int m, int n,
                             COO *sparse)
{
    int i, j, NZ;
    COO sp;
    NZ = 0;
    /* Figure out how many nonzeros we're going to have. */
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            double val = dense[j*m + i];
            if (fabs(val) > 1e-15) {
                NZ++;
            }
        }
    }
    alloc_sparse(m, n, NZ, &sp);

    NZ = 0;
    /* Fill up the sparse matrix */
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            double val = dense[j*m + i];
            if (fabs(val) > 1e-15) {
                sp->coords[NZ].i = i;
                sp->coords[NZ].j = j;
                sp->data[NZ] = val;
                NZ++;
            }
        }
    }
    *sparse = sp;
}

/*
 * Read a sparse matrix from a file.
 *
 * file - The filename to read
 * sparse - The newly read sparse matrix (allocated here)
 */
void read_sparse(const char *file, COO *sparse)
{
    COO sp;
    int i, j, k, m, n, NZ;
    double val;
    int c;
    FILE *f = fopen(file, "r");
    if (!f) {
        fprintf(stderr, "Unable to open %s for reading.\n", file);
        exit(1);
    }
    c = fscanf(f, "%d %d %d\n", &m, &n, &NZ);
    if (c != 3) {
        fprintf(stderr, "File format incorrect on line 1, expecting 3 integers, got %d\n", c);
        fclose(f);
        exit(1);
    }
    if (NZ > m*n) {
        fprintf(stderr, "More nonzeros (%d) than matrix entries (%d x %d)!\n", NZ, m, n);
        fclose(f);
        exit(1);
    }
    alloc_sparse(m, n, NZ, &sp);
    k = 0;
    while ((c = fscanf(f, "%d %d %lg\n", &i, &j, &val)) == 3) {
        if (k >= NZ) {
            fprintf(stderr, "File has nonzero lines than expected (%d)\n", NZ);
            fclose(f);
            free_sparse(&sp);
            exit(1);
        }
        if (i >= m || j >= n) {
            fprintf(stderr, "Entry on line %d incorrect, index (%d, %d) out of bounds for %d x %d matrix\n", k + 2, i, j, m, n);
            fclose(f);
            free_sparse(&sp);
            exit(1);
        }
        sp->coords[k].i = i;
        sp->coords[k].j = j;
        sp->data[k] = val;
        k++;
    }

    if (k != NZ) {
        fprintf(stderr, "File has fewer lines (%d) than expected (%d)\n",
                k, NZ);
        fclose(f);
        free_sparse(&sp);
        exit(1);
    }
    *sparse = sp;
    fclose(f);
}

/*
 * Write a sparse matrix to a file.
 *
 * f - The file handle.
 * sp - The sparse matrix to write.
 */
void write_sparse(FILE *f, COO sp)
{
    int i;
    fprintf(f, "%d %d %d\n", sp->m, sp->n, sp->NZ);
    for (i = 0; i < sp->NZ; i++) {
        fprintf(f, "%d %d %g\n", sp->coords[i].i, sp->coords[i].j, sp->data[i]);
    }
}

/*
 * Print a sparse matrix to stdout
 *
 * sp - The sparse matrix to print.
 */
void print_sparse(COO sp)
{
    write_sparse(stdout, sp);
}
