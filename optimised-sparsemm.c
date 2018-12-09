#include "utils.h"
#include <stdlib.h>

// taken from https://stackoverflow.com/questions/37538/how-do-i-determine-the-size-of-my-array-in-c 
#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))

void basic_sparsemm(const COO, const COO, COO *);
void basic_sparsemm_sum(const COO, const COO, const COO,
                        const COO, const COO, const COO,
                        COO *);

/* Computes C = A*B.
 * C should be allocated by this routine.
 */
void optimised_sparsemm(const COO A, const COO B, COO *C)
{
    COO BT, res, Tres;
    int apos, btpos, cpos, 
    ANZ, BTNZ, arow, acol, btrow, btcol;
    double adata, btdata;

    ANZ = A->NZ;
    BTNZ = B->NZ;

    print_sparse(A);
    print_sparse(B);

    // bound to the multiplication matrixx https://www.degruyter.com/downloadpdf/j/comp.2014.4.issue-1/s13537-014-0201-x/s13537-014-0201-x.pdf
    alloc_sparse(A->m, B->n, ANZ * BTNZ, &res);

    // set method to intialize res.data to zeros
    transpose_coo(B, &BT);

    cpos = 0;

    for(apos = 0; apos < ANZ; apos++) {
        arow = A->coords[apos].i;
        acol = A->coords[apos].j;
        adata = A->data[apos];
        for(btpos = 0; btpos < BTNZ; btpos++) {
            btrow = BT->coords[btpos].i;
            btcol = BT->coords[btpos].j;
            btdata = BT->data[btpos];
            // transpose is not really needed for this case
            if (acol == btcol) {
                res->coords[cpos].i = arow;
                res->coords[cpos].j = btrow;
                res->data[cpos] =  adata * btdata;
                cpos++;
            }
        }
    }

    res->NZ = cpos;
    res->coords = realloc(res->coords, cpos * sizeof(struct coord));
    res->data = realloc(res->data, cpos * sizeof(double));

    transpose_coo(res, &Tres);
    print_sparse(res);
    print_sparse(Tres);

    free_sparse(&res);
    free_sparse(&BT);
    free_sparse(&Tres);

    return basic_sparsemm(A, B, C);
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

// http://delivery.acm.org/10.1145/360000/355796/p250-gustavson.pdf?ip=129.234.0.23&id=355796&acc=ACTIVE%20SERVICE&key=BF07A2EE685417C5%2EAB9A2A9F43EF7438%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1544294337_058d03490eaa8bf2d29373ed515d88b5 
// transpose algo for CSR p8
// Algorithm to rezero Boolean array xb p11
// multiplication algo p13