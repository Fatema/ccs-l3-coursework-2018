#include "utils.h"
#include <stdlib.h>

void coo_sum_duplicates(const COO coo, COO *nodups);

void transpose_coo_acc(const COO coo, COO *transposed);

void csr_transpose(const CSR csr, CSR *transposed);

void coo2csr_mm_multiply(const COO a, const COO b, COO *c);

void csr_mm_multiply_acc(const COO a, const COO b, COO *c);

void coo_mm_multiply(const COO a, const COO b, COO *c);

void csr_mm_multiply(const CSR a, const CSR b, CSR *c);

void coo_mm_multiply_acc(const COO a, const COO b, COO *c);

void csr_sum(const CSR A, const CSR B, CSR *sum);

void coo2csr_mm_multiply_sum(const COO A, const COO B, const COO C,
                             const COO D, const COO E, const COO F,
                             COO *O);

void basic_sparsemm(const COO, const COO, COO *);

void basic_sparsemm_sum(const COO, const COO, const COO,
                        const COO, const COO, const COO,
                        COO *);

/* Computes C = A*B.
 * C should be allocated by this routine.
 * this is based on https://www.geeksforgeeks.org/operations-sparse-matrices/ 
 */
void optimised_sparsemm(const COO A, const COO B, COO *C) {
    return coo2csr_mm_multiply(A, B, C);
}

/* Computes O = (A + B + C) (D + E + F).
 * O should be allocated by this routine.
 */
void optimised_sparsemm_sum(const COO A, const COO B, const COO C,
                            const COO D, const COO E, const COO F,
                            COO *O) {
    return coo2csr_mm_multiply_sum(A, B, C, D, E, F, O);
}


void coo_mm_multiply_acc(const COO A, const COO B, COO *C) {
    COO res, Tres;
    int apos, bpos, cpos,
            ANZ, BNZ, arow, acol, brow, bcol;
    double adata, bdata;

    struct coord *acoord, *bcoord, *rcoord;
    double *adataarr, *bdataarr, *rdataarr;

    acoord = A->coords;
    bcoord = B->coords;

    adataarr = A->data;
    bdataarr = B->data;

    ANZ = A->NZ;
    BNZ = B->NZ;

    // bound to the multiplication matrixx https://www.degruyter.com/downloadpdf/j/comp.2014.4.issue-1/s13537-014-0201-x/s13537-014-0201-x.pdf
    alloc_sparse(A->m, B->n, ANZ * BNZ, &res);

    rcoord = res->coords;
    rdataarr = res->data;

    cpos = -1;

    // with this approach sorting A and B is not really necessary, neither is transposing B
#pragma acc data copyin(acoord[0:ANZ], bcoord[0:BNZ], adataarr[0:ANZ], bdataarr[0:BNZ]), copyout(rcoord[0:ANZ * BNZ], rdataarr[0:ANZ * BNZ])
#pragma acc parallel loop
    for (apos = 0; apos < ANZ; apos++) {
        arow = acoord[apos].i;
        acol = acoord[apos].j;
        adata = adataarr[apos];
#pragma acc loop
        for (bpos = 0; bpos < BNZ; bpos++) {
            brow = bcoord[bpos].j;
            bcol = bcoord[bpos].i;
            bdata = bdataarr[bpos];
            // transpose is not really needed for this case
            if (acol == bcol) {
                // this is slowing down the parallization
#pragma acc atomic update
                cpos++; // I can use this to keep track if I'm about to run out of allocated memory
                rcoord[cpos].i = arow;
                rcoord[cpos].j = brow;
                rdataarr[cpos] = adata * bdata;
            }
        }
    }

    cpos++;

    res->NZ = cpos;
    res->coords = realloc(rcoord, cpos * sizeof(struct coord));
    res->data = realloc(rdataarr, cpos * sizeof(double));

    // the removing of duplicates can be done in merge sort style
    // sort the result
    transpose_coo_acc(res, &Tres);
    transpose_coo_acc(Tres, &res);

    coo_sum_duplicates(res, C);

    free_sparse(&res);
    free_sparse(&Tres);
}

void coo_mm_multiply(const COO A, const COO B, COO *C) {
    COO res, Tres;
    int apos, bpos, cpos,
            ANZ, BNZ, arow, acol, brow, bcol;
    double adata, bdata;

    struct coord *acoord, *bcoord, *rcoord;
    double *adataarr, *bdataarr, *rdataarr;

    acoord = A->coords;
    bcoord = B->coords;

    adataarr = A->data;
    bdataarr = B->data;

    ANZ = A->NZ;
    BNZ = B->NZ;

    // bound to the multiplication matrixx https://www.degruyter.com/downloadpdf/j/comp.2014.4.issue-1/s13537-014-0201-x/s13537-014-0201-x.pdf
    alloc_sparse(A->m, B->n, ANZ * BNZ, &res);

    rcoord = res->coords;
    rdataarr = res->data;

    cpos = -1;

    for (apos = 0; apos < ANZ; apos++) {
        arow = acoord[apos].i;
        acol = acoord[apos].j;
        adata = adataarr[apos];
        for (bpos = 0; bpos < BNZ; bpos++) {
            brow = bcoord[bpos].j;
            bcol = bcoord[bpos].i;
            bdata = bdataarr[bpos];
            // transpose is not really needed for this case
            if (acol == bcol) {
                // this is slowing down the parallization
                cpos++; // I can use this to keep track if I'm about to run out of allocated memory
                rcoord[cpos].i = arow;
                rcoord[cpos].j = brow;
                rdataarr[cpos] = adata * bdata;
            }
        }
    }

    cpos++;

    res->NZ = cpos;
    res->coords = realloc(rcoord, cpos * sizeof(struct coord));
    res->data = realloc(rdataarr, cpos * sizeof(double));

    // the removing of duplicates can be done in merge sort style
    // sort the result
    transpose_coo_acc(res, &Tres);
    transpose_coo_acc(Tres, &res);

    coo_sum_duplicates(res, C);

    free_sparse(&res);
    free_sparse(&Tres);
}


/*
 * Transpose a coo format sparse matrix
 * This is based on https://www.geeksforgeeks.org/operations-sparse-matrices/ 
 *
 * coo - sparse matrix
 * transposed - output transposed coo format sparse matrix (allocated by this routine)
 */
void transpose_coo_acc(const COO coo, COO *transposed) {
    // set n to number of rows
    int n = coo->m;
    // set m to number of columns
    int m = coo->n;
    int NZ = coo->NZ;
    int i;

    struct coord *coords = coo->coords;
    double *coodata = coo->data;

    COO sp;

    alloc_sparse(m, n, NZ, &sp);

    struct coord *spcoords = sp->coords;
    double *spdata = sp->data;

    // count number of elements in each column
    int count[m];
    for (i = 0; i < m; i++) {
        count[i] = 0;
    }

    for (i = 0; i < NZ; i++) {
        count[coords[i].j]++;
    }

    // to count number of elements having col smaller 
    // than particular i 
    int index[m];

    // as there is no col with value < 1 
    index[0] = 0;

    // initialize rest of the indices
    // can be done with Parallel Prefix Sum
    for (i = 1; i < m; i++) {
        index[i] = index[i - 1] + count[i - 1];
    }

    int rpos;

    // this one cannot be easily parallized 
    for (i = 0; i < NZ; i++) {
        rpos = index[coords[i].j];
        spcoords[rpos].i = coords[i].j;
        spcoords[rpos].j = coords[i].i;
        spdata[rpos] = coodata[i];
        index[coords[i].j]++;
    }

    sp->coords = spcoords;
    sp->data = spdata;

    // the above method ensures 
    // sorting of transpose matrix 
    // according to row-col value
    *transposed = sp;
}

void coo_sum_duplicates(const COO coo, COO *nodups) {
    COO sp;
    int NZ = coo->NZ;
    int i;

    alloc_sparse(coo->m, coo->n, NZ, &sp);

    int nnz = 0;
    sp->coords[0] = coo->coords[0];
    sp->data[0] = coo->data[0];

    for (i = 1; i < NZ; i++) {
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

void csr_transpose(const CSR csr, CSR *transposed) {
    int i, j, ir, m, n, NZ, nir, jpt, jp, q;
    CSR csrt;

    m = csr->m; // in csr the number of rows is + 1
    n = csr->n;
    NZ = csr->NZ;

    alloc_sparse_csr(n, m, NZ, &csrt);

    // count the number of columns (rows for the transposed csr)
    for (i = 0; i < NZ; i++) {
        csrt->I[csr->J[i] + 1]++;
    }

    // set a pointer for the cumulative sum of the rows index
    // this is used to shift the cumulative sum of the transposed row index
    int p[n + 1];
    q = 0;
    for (i = 0; i < n + 1; i++) {
        p[i] = q;
        q += csrt->I[i];
        csrt->I[i] = p[i];
    }

    for (i = 0; i < m + 1; i++) {
        ir = csr->I[i];
        nir = csr->I[i + 1];
        for (jp = ir; jp < nir; jp++) {
            j = csr->J[jp];
            jpt = csrt->I[j + 1];
            csrt->J[jpt] = i;
            csrt->data[jpt] = csr->data[jp];
            csrt->I[j + 1] = jpt + 1;
        }
    }

    csrt->I[0] = 0;

    *transposed = csrt;
}

void coo2csr_mm_multiply(const COO acoo, const COO bcoo, COO *c) {
    CSR a, b, ctemp;

    printf("converting %d", 1);
    convert_coo_to_csr(acoo, &a);
    convert_coo_to_csr(bcoo, &b);

    printf("doing the multiplicaton %d", 2);
    csr_mm_multiply(a, b, &ctemp);

    printf("converting %d", 3);
    convert_csr_to_coo(ctemp, c);
}

void coo2csr_mm_multiply_sum(const COO A, const COO B, const COO C,
                             const COO D, const COO E, const COO F,
                             COO *O) {

    CSR acsr, bcsr, ccsr, dcsr, ecsr, fcsr;

    CSR otemp, abc, def; // temporary matrix to hold o data and the sum results

    // convert the matrices to csr format (rows will be sorted after the conversion)
    printf("converting %d", 1);
    convert_coo_to_csr(A, &acsr);
    convert_coo_to_csr(B, &bcsr);
    convert_coo_to_csr(C, &ccsr);
    convert_coo_to_csr(D, &dcsr);
    convert_coo_to_csr(E, &ecsr);
    convert_coo_to_csr(F, &fcsr);

    // the sum depends on the matrices being fully sorted so use transpose to sort the columns
    // transposing won't affect the result of the sum
    // (it could affect the parallelisation of the sum process)
    printf("transposing %d", 2);
    csr_transpose(acsr, &acsr);
    csr_transpose(bcsr, &bcsr);
    csr_transpose(ccsr, &ccsr);
    csr_transpose(dcsr, &dcsr);
    csr_transpose(ecsr, &ecsr);
    csr_transpose(fcsr, &fcsr);

    printf("doing the sum %d", 3);
    csr_sum(acsr, bcsr, &abc);
    csr_sum(abc, ccsr, &abc);

    csr_sum(dcsr, ecsr, &def);
    csr_sum(def, fcsr, &def);

    // transpose the sums to be rows oriented
    csr_transpose(abc, &abc);
    csr_transpose(def, &def);

    printf("doing the multiplicaton %d", 4);
    csr_mm_multiply(abc, def, &otemp);

    convert_csr_to_coo(otemp, O);

    printf("freeing memory %d", 5);
    free_sparse_csr(&acsr);
    free_sparse_csr(&bcsr);
    free_sparse_csr(&ccsr);
    free_sparse_csr(&dcsr);
    free_sparse_csr(&ecsr);
    free_sparse_csr(&fcsr);
    free_sparse_csr(&abc);
    free_sparse_csr(&def);
    free_sparse_csr(&otemp);
}

void csr_sum(const CSR A, const CSR B, CSR *sum) {
    CSR temp;

    int m, n, i, ai, nai, bi, nbi, ncol, j;

    m = A->m;
    n = A->n;

    alloc_sparse_csr(m, n, A->NZ + B->NZ, &temp);

    ncol = 0;
    for (i = 0; i < m; i++) {
        ai = A->I[i];
        nai = A->I[i + 1];

        bi = B->I[i];
        nbi = B->I[i + 1];

        while (ai < nai && bi < nbi) {
            if (A->J[ai] < B->J[bi]) {
                temp->J[ncol] = A->J[ai];
                temp->data[ncol] = A->data[ai];
                ncol++;
                ai++;
            } else if (A->J[ai] == B->J[bi]) {
                temp->J[ncol] = A->J[ai];
                temp->data[ncol] = A->data[ai] + B->data[bi];
                ncol++;
                ai++;
                bi++;
            } else {
                temp->J[ncol] = B->J[bi];
                temp->data[ncol] = B->data[bi];
                ncol++;
                bi++;
            }
        }

        // add the remaining elements
        for(j = ai; j < nai; j++){
            temp->J[ncol] = A->J[j];
            temp->data[ncol] = A->data[j];
            ncol++;
        }

        // add the remaining elements
        for(j = bi; j < nbi; j++){
            temp->J[ncol] = B->J[j];
            temp->data[ncol] = B->data[j];
            ncol++;
        }

        temp->I[i + 1] = ncol;
    }

    temp->J = realloc(temp->J, ncol * sizeof(int));
    temp->data = realloc(temp->data, ncol * sizeof(double));

    temp->NZ = ncol;

    *sum = temp;
}

// there is something wrong here needs to be traced
void csr_mm_multiply(const CSR a, const CSR b, CSR *c) {
    CSR ctemp; // temporary matrix to hold c data

    int ip, i, jp, j, kp, k, vp, v; // scalar intergers
    int r, p, q; // matrices size
    r = b->n; // b = q x r
    p = a->m; // a = p x q
    q = a->n; // must be equal to b->m

    // vectors of length r that will hold integer and floating point data
    int xb[r];
    double x[r];

    // max value for ctemp size
    int ibot = p * r;

    alloc_sparse_csr(p, r, ibot, &ctemp);

    ip = 0; // keeps track of value positions for matrix c


    for (v = 0; v < r; v++) {
        xb[v] = -1;
        x[v] = -1;
    }

    for (i = 0; i < p + 1; i++) {
        ctemp->I[i] = ip;
        for (jp = a->I[i]; jp < a->I[i + 1]; jp++) {
            j = a->J[jp];
            for (kp = b->I[j]; kp < b->I[j + 1]; kp++) {
                k = b->J[kp];
                if (xb[k] != i) {
                    ctemp->J[ip] = k;
                    ip++;
                    xb[k] = i;
                    x[k] = a->data[jp] * b->data[kp];
                } else {
                    x[k] += a->data[jp] * b->data[kp];
                }
            }
        }

        for (vp = ctemp->I[i]; vp < ip; vp++) {
            v = ctemp->J[vp];
            ctemp->data[vp] = x[v];
        }
    }

    ctemp->I[p + 1] = ip;
    ctemp->J = realloc(ctemp->J, ip * sizeof(int));
    ctemp->data = realloc(ctemp->data, ip * sizeof(double));
    ctemp->NZ = ip;

    *c = ctemp;
}