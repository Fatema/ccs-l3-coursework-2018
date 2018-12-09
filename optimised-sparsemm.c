#include "utils.h"
#include <stdlib.h>

void coo_sum_duplicates(const COO coo, COO *nodups);
void transpose_coo(const COO coo, COO *transposed);
void basic_sparsemm(const COO, const COO, COO *);
void basic_sparsemm_sum(const COO, const COO, const COO,
                        const COO, const COO, const COO,
                        COO *);

/* Computes C = A*B.
 * C should be allocated by this routine.
 * this is based on https://www.geeksforgeeks.org/operations-sparse-matrices/ 
 */
void optimised_sparsemm(const COO A, const COO B, COO *C)
{
    COO res, Tres;
    int apos, bpos, cpos, 
    ANZ, BNZ, arow, acol, brow, bcol;
    double adata, bdata;

    ANZ = A->NZ;
    BNZ = B->NZ;

    // bound to the multiplication matrixx https://www.degruyter.com/downloadpdf/j/comp.2014.4.issue-1/s13537-014-0201-x/s13537-014-0201-x.pdf
    alloc_sparse(A->m, B->n, ANZ * BNZ, &res);

    cpos = 0;

    // with this approach sorting A and B is not really necessary, neither is transposing B 
    for(apos = 0; apos < ANZ; apos++) {
        arow = A->coords[apos].i;
        acol = A->coords[apos].j;
        adata = A->data[apos];
        for(bpos = 0; bpos < BNZ; bpos++) {
            brow = B->coords[bpos].j;
            bcol = B->coords[bpos].i;
            bdata = B->data[bpos];
            // transpose is not really needed for this case
            if (acol == bcol) {
                res->coords[cpos].i = arow;
                res->coords[cpos].j = brow;
                res->data[cpos] =  adata * bdata;
                cpos++;
            }
        }
    }

    res->NZ = cpos;
    res->coords = realloc(res->coords, cpos * sizeof(struct coord));
    res->data = realloc(res->data, cpos * sizeof(double));

    // the removing of duplicates can be done in merge sort style
    // sort the result
    transpose_coo(res, &Tres);
    transpose_coo(Tres, &res);

    coo_sum_duplicates(res, C);

    free_sparse(&res);
    free_sparse(&Tres);
}

/* Computes O = (A + B + C) (D + E + F).
 * O should be allocated by this routine.
 */
void optimised_sparsemm_sum(const COO A, const COO B, const COO C,
                            const COO D, const COO E, const COO F,
                            COO *O)
{
    
    return basic_sparsemm_sum(A, B, C, D, E, F, O);
}


/*
 * Transpose a coo format sparse matrix
 * This is based on https://www.geeksforgeeks.org/operations-sparse-matrices/ 
 *
 * coo - sparse matrix
 * transposed - output transposed coo format sparse matrix (allocated by this routine)
 */
void transpose_coo(const COO coo, COO *transposed)
{
    // set n to number of rows
    int n = coo->m;
    // set m to number of columns
    int m = coo->n;
    int NZ = coo->NZ;
    int i;

    COO sp;

    alloc_sparse(m, n, NZ, &sp);

    // count number of elements in each column
    int count[m];
    for(i = 0; i < m; i++){
        count[i] = 0;
    }

    // this is easily parallized 
    for(i = 0; i < NZ; i++){
        count[coo->coords[i].j]++;
    }

    // to count number of elements having col smaller 
    // than particular i 
    int index[m];

    // as there is no col with value < 1 
    index[0] = 0;

    // initialize rest of the indices
    // can be done with Parallel Prefix Sum
    for(i = 1; i < m; i++){
        index[i] = index[i - 1] + count[i - 1];
    }
    
    int rpos;

    // this one cannot be easily parallized 
    for (i = 0; i < NZ; i++){
        rpos = index[coo->coords[i].j];
        sp->coords[rpos].i = coo->coords[i].j;
        sp->coords[rpos].j = coo->coords[i].i;
        sp->data[rpos] = coo->data[i];
        index[coo->coords[i].j]++;
    }
    
    // the above method ensures 
    // sorting of transpose matrix 
    // according to row-col value
    *transposed = sp;
}

void coo_sum_duplicates(const COO coo, COO *nodups){
    COO sp;
    int NZ = coo->NZ;
    int i;

    alloc_sparse(coo->m, coo->n, NZ, &sp);
    
    int nnz = 0;
    sp->coords[0] = coo->coords[0];
    sp->data[0] = coo->data[0];

    for(i = 1; i < NZ; i++){
        if (coo->coords[i].i == sp->coords[nnz].i && coo->coords[i].j == sp->coords[nnz].j) {
            sp->data[nnz] += coo->data[i];
        } else {
            nnz++;
            sp->coords[nnz] = coo->coords[i];
            sp->data[nnz] = coo->data[i];
        }
    }

    // since nnz start at zero at the end it must be incremented by 1 to get the actual size
    nnz++;

    sp->NZ = nnz;
    sp->coords = realloc(sp->coords, nnz * sizeof(struct coord));
    sp->data = realloc(sp->data, nnz * sizeof(double));

    *nodups = sp;
}

// http://delivery.acm.org/10.1145/360000/355796/p250-gustavson.pdf?ip=129.234.0.23&id=355796&acc=ACTIVE%20SERVICE&key=BF07A2EE685417C5%2EAB9A2A9F43EF7438%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1544294337_058d03490eaa8bf2d29373ed515d88b5 
// transpose algo for CSR p8
// Algorithm to rezero Boolean array xb p11
// multiplication algo p13