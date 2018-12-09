/**
**
** Routines to read/write matrix.
**
** ejim  Wed Mar  4 15:16:14 PST 1998
**
**/

#include <stdio.h>

void csr2csc(int n, int m, int nz, double *a, int *col_idx, int *row_start,
             double *csc_a, int *row_idx, int *col_start);
void coo2csr_in(int n, int nz, double *a, int *i_idx, int *j_idx);
void coo2csr(int n, int nz, double *a, int *i_idx, int *j_idx,
	     double *csr_a, int *col_idx, int *row_start);


/* write CSR format */
/* 1st line : % number_of_rows number_of_columns number_of_nonzeros
   2nd line : % base of index 
   3rd line : row_number  nz_r(=number_of_nonzeros_in_the_row)
   next nz_r lines : column_index value(when a != NULL)
   next line : row_number  nz_r(=number_of_nonzeros_in_the_row)
   next nz_r lines : column_index value(when a != NULL)
   ...
   */

void write_csr (char *fn, int m, int n, int nz,
		int *row_start, int *col_idx, double *a)
{
  FILE *f;
  int i, j;

  if ((f = fopen(fn, "w")) == NULL){ 
    printf ("can't open file <%s> \n", fn);
    exit(1);
  }

  fprintf (f, "%s %d %d %d\n", "%", m, n, nz);
  fprintf (f, "%s 0\n", "%");

  for (i=0; i<m; i++){
    fprintf(f, "%d %d\n", i, row_start[i+1]-row_start[i]);

    for (j=row_start[i]; j<row_start[i+1]; j++){
      if (a)
	fprintf(f, "%d %20.19g\n", col_idx[j], a[j]);
      else
	fprintf(f, "%d\n", col_idx[j]);
    }
  }

  fclose (f);

}


/* reads matrix market format (coordinate) and returns
   csr format */

void read_mm_matrix (char *fn, int *m, int *n, int *nz,
		  int **i_idx, int **j_idx, double **a)
{
  MM_typecode matcode;
  FILE *f;
  int i,k;
  int base;

  if ((f = fopen(fn, "r")) == NULL) {
    printf ("can't open file <%s> \n", fn);
    exit(1);
  }
  if (mm_read_banner(f, &matcode) != 0){
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (! (mm_is_matrix(matcode) && mm_is_sparse(matcode)) ){
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  /* find out size of sparse matrix .... */

  fscanf(f, "%*s %d %d %d", m, n, nz);    

  fscanf(f, "%*s %d", &base);     /* indentifies starting index */

  /* reserve memory for matrices */
  if (mm_is_symmetric(matcode)){
    *i_idx = (int *) my_malloc(*nz *2 * sizeof(int));
    *j_idx = (int *) my_malloc(*nz *2 * sizeof(int));
    *a = (double *) my_malloc(*nz *2 * sizeof(double));
  }
  else {
    *i_idx = (int *) my_malloc(*nz * sizeof(int));
    *j_idx = (int *) my_malloc(*nz * sizeof(int));
    *a = (double *) my_malloc(*nz * sizeof(double));
  }

  if (!(*i_idx) || !(*j_idx) || !(*a)){
    printf ("cannot allocate memory for %d, %d, %d sparse matrix\n", *m, *n, *nz);
    exit(1);
  }


  k=0;
  for (i=0; i<*nz; i++)  {
    if (mm_is_pattern(matcode)){
      fscanf(f, "%d %d", &(*i_idx)[i], &(*j_idx)[i]);
      (*i_idx)[i] -= base;  /* adjust from 1-based to 0-based */
      (*j_idx)[i] -= base;

      (*a)[i] = random_double(-1, 1);
    }
    else if (mm_is_real(matcode)){
      fscanf(f, "%d %d %lg", &(*i_idx)[i], &(*j_idx)[i], &(*a)[i]);
      (*i_idx)[i] -= base;  /* adjust from 1-based to 0-based */
      (*j_idx)[i] -= base;
    }

    if (mm_is_symmetric(matcode)){
      if ( (*i_idx)[i] != (*j_idx)[i] ){
	(*i_idx)[*nz+k] = (*j_idx)[i];
	(*j_idx)[*nz+k] = (*i_idx)[i];
	(*a)[*nz+k] = (*a)[i];
	k++;
      }
    }
  }
  *nz += k;

  fclose(f);

  coo2csr_in (*m, *nz, *a, *i_idx, *j_idx);
}

/* reads Harwell-Boeing format matrix and returns CSR format */

#define LINE_LEN     90
#define INPUT_WIDTH  80

void read_hb_matrix (char *fn, int *m, int *n, int *nz,
		     int **row_start, int **col_idx, double **a)
{
  MM_typecode matcode;
  FILE *f;
  int i, field, j, k;
  int row, rhs;
  char buffer[LINE_LEN];
  char mat_format[4];
  int *i_idx, *j_idx;
  int *col_start, *row_idx;
  double *coo_a;
  char    start_format[INPUT_WIDTH];
  char    idx_format[INPUT_WIDTH];
  int start_input_repeat, start_input_width;
  int idx_input_repeat, idx_input_width;
  int start_lines, idx_lines, l;

  if ((f = fopen(fn, "r")) == NULL) {
    printf ("can't open file <%s> \n", fn);
    exit(1);
  }
  fgets (buffer, LINE_LEN, f); /* title line */
  fgets (buffer, LINE_LEN, f); /* line counts */
  sscanf(buffer, "%*d %d %d %*d %d", &start_lines, &idx_lines, &rhs);
  fgets (buffer, LINE_LEN, f); /* elements counts */
  sscanf(buffer, "%s %d %d %d", mat_format, m, n, nz);
  printf("type=%s m=%d n=%d nz=%d rhs=%d\n", mat_format, *m, *n, *nz, rhs);
  fgets (buffer, LINE_LEN, f); /* formats */
  sscanf(buffer, "%s %s", start_format, idx_format);
  sscanf(start_format, "(%d%*c%d)", &start_input_repeat, &start_input_width);
  sscanf(idx_format, "(%d%*c%d)", &idx_input_repeat, &idx_input_width);
  printf("%d start_input lines %d repeats %d: %d col_input lines %d repeats %d\n",
	 start_lines, start_input_width, start_input_repeat,
	 idx_lines, idx_input_width, idx_input_repeat);

  if (rhs)
    fgets (buffer, LINE_LEN, f); /* rhs header */

  mm_clear_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_sparserow(&matcode);

  switch (mat_format[0]){
  case 'r':
  case 'R':
    mm_set_real(&matcode);
    break;
  case 'c':
  case 'C':
    mm_set_complex(&matcode);
    break;
  case 'p':
  case 'P':
    mm_set_pattern(&matcode);
    break;
  default:
    printf ("<%s> is not Harwell Boeing matrix format %s\n", fn, mat_format);
    exit (0);
  }

  switch (mat_format[1]){
  case 'u':
  case 'U':
  case 'r':
  case 'R':
    mm_set_general(&matcode);
    break;
  case 's':
  case 'S':
    mm_set_symmetric(&matcode);
    break;
  case 'h':
  case 'H':
    mm_set_hermitian(&matcode);
    break;
  case 'z':
  case 'Z':
    mm_set_skew(&matcode);
    break;
  default:
    printf ("<%s> is not Harwell Boeing matrix format %s\n", fn, mat_format);
    exit (0);
  }

  switch (mat_format[2]){
  case 'a':
  case 'A':
    break;
  case 'u':
  case 'U':
    printf ("<%s> is not unassembled Harwell Boeing matrix format %s\n", 
	    fn, mat_format);
    exit (0);
  default:
    printf ("<%s> is not Harwell Boeing matrix format %s\n", fn, mat_format);
    exit (0);
  }
    
    
  *row_start = (int *) my_malloc((*m+1) * sizeof(int));

  /* reserve memory for matrices */
  if (mm_is_symmetric(matcode)){
    i_idx = (int *) my_malloc(*nz *2 * sizeof(int));
    j_idx = (int *) my_malloc(*nz *2 * sizeof(int));
    coo_a = (double *) my_malloc(*nz *2 * sizeof(double));
  }
  else {
    col_start = (int *) my_malloc((*n+1) * sizeof(int));
    row_idx = (int *) my_malloc(*nz * sizeof(int));

    *col_idx = (int *) my_malloc(*nz * sizeof(int));
    *a = (double *) my_malloc(*nz * sizeof(double));
  }


  if (mm_is_symmetric(matcode)){
    for (i=l=0; l<start_lines; l++){ 
      fgets (buffer, LINE_LEN, f); 
      for (field=0; (field<start_input_repeat) && (i<=*m); field++){
	for(j=0; j<start_input_width; j++)
	  start_format[j] = buffer[field*start_input_width+j];
	start_format[j] =  0;

	(*row_start)[i++] = atoi(start_format)-1;
      }
    }

    for (i=l=k=row=0; l<idx_lines; l++){ 
      fgets (buffer, LINE_LEN, f); 
      for (field=0; (field<idx_input_repeat) && (i<*nz); field++, i++){
	for(j=0; j<idx_input_width; j++)
	  idx_format[j] = buffer[field*idx_input_width+j];
	idx_format[j] =  0;

	if (i==(*row_start)[row+1]) row++;
	i_idx[i] = row;
	j_idx[i] = atoi(idx_format)-1;

	coo_a[i] = random_double(-1, 1);

	if ( i_idx[i] != j_idx[i] ){
	  i_idx[*nz+k] = j_idx[i];
	  j_idx[*nz+k] = i_idx[i];
	  coo_a[*nz+k] = coo_a[i];
	  k++;
	}
      }
    }
    *nz += k;

    *col_idx = (int *) my_malloc(*nz * sizeof(int));
    *a = (double *) my_malloc(*nz * sizeof(double));
    coo2csr (*m, *nz, coo_a, i_idx, j_idx, *a, *col_idx, *row_start);

    free (i_idx);
    free (j_idx);
    free (coo_a);

  }
  else {
    for (i=l=0; l<start_lines; l++){ 
      fgets (buffer, LINE_LEN, f); 
      for (field=0; (field<start_input_repeat) && (i<=*n); field++){
	for(j=0; j<start_input_width; j++)
	  start_format[j] = buffer[field*start_input_width+j];
	start_format[j] =  0;

	col_start[i++] = atoi(start_format)-1;
      }
    }


    for (i=l=0; l<idx_lines; l++){ 
      fgets (buffer, LINE_LEN, f); 
      for (field=0; (field<idx_input_repeat) && (i<*nz); field++, i++){
	for(j=0; j<idx_input_width; j++)
	  idx_format[j] = buffer[field*idx_input_width+j];
	idx_format[j] =  0;

	row_idx[i] = atoi(idx_format)-1;

	(*a)[i] = random_double(-1, 1);
      }
    }

    csr2csc(*n, *m, *nz, NULL, row_idx, col_start, NULL, *col_idx, *row_start);


    free (row_idx);
    free (col_start);
  }

  fclose(f);
}


void sort(int *col_idx, double *a, int start, int end)
{
  int i, j, it;
  double dt;

  for (i=end-1; i>start; i--)
    for(j=start; j<i; j++)
      if (col_idx[j] > col_idx[j+1]){

	if (a){
	  dt=a[j]; 
	  a[j]=a[j+1]; 
	  a[j+1]=dt;
        }
	it=col_idx[j]; 
	col_idx[j]=col_idx[j+1]; 
	col_idx[j+1]=it;
	  
      }
}



/* converts COO format to CSR format, in-place,
   if SORT_IN_ROW is defined, each row is sorted in column index.
   On return, i_idx contains row_start position */

void coo2csr_in(int n, int nz, double *a, int *i_idx, int *j_idx)
{
  int *row_start;
  int i, j;
  int init, i_next, j_next, i_pos;
  double dt, a_next;

  row_start = (int *)malloc((n+1)*sizeof(int));
  if (!row_start){
    printf ("coo2csr_in: cannot allocate temporary memory\n");
    exit (1);
  }
  for (i=0; i<=n; i++) row_start[i] = 0;

  /* determine row lengths */
  for (i=0; i<nz; i++) row_start[i_idx[i]+1]++;

  for (i=0; i<n; i++) row_start[i+1] += row_start[i];

  for (init=0; init<nz; ){
    dt = a[init];
    i = i_idx[init];
    j = j_idx[init];
    i_idx[init] = -1;
    while (1){
      i_pos = row_start[i];
      a_next = a[i_pos];
      i_next = i_idx[i_pos];
      j_next = j_idx[i_pos];

      a[i_pos] = dt;
      j_idx[i_pos] = j;
      i_idx[i_pos] = -1;
      row_start[i]++;
      if (i_next < 0) break;
      dt = a_next;
      i = i_next;
      j = j_next;

    }
    init++;
    while ((i_idx[init] < 0) && (init < nz))  init++;
  }


  /* shift back row_start */
  for (i=0; i<n; i++) i_idx[i+1] = row_start[i];
  i_idx[0] = 0;


  for (i=0; i<n; i++){
    sort (j_idx, a, i_idx[i], i_idx[i+1]);
  }

}

/* converts COO format to CSR format, not in-place,
   if SORT_IN_ROW is defined, each row is sorted in column index */


void coo2csr(int n, int nz, double *a, int *i_idx, int *j_idx,
	     double *csr_a, int *col_idx, int *row_start)
{
  int i, l;

  for (i=0; i<=n; i++) row_start[i] = 0;

  /* determine row lengths */
  for (i=0; i<nz; i++) row_start[i_idx[i]+1]++;


  for (i=0; i<n; i++) row_start[i+1] += row_start[i];


  /* go through the structure  once more. Fill in output matrix. */
  for (l=0; l<nz; l++){
    i = row_start[i_idx[l]];
    csr_a[i] = a[l];
    col_idx[i] = j_idx[l];
    row_start[i_idx[l]]++;
  }

  /* shift back row_start */
  for (i=n; i>0; i--) row_start[i] = row_start[i-1];

  row_start[0] = 0;

  for (i=0; i<n; i++){
    sort (col_idx, csr_a, row_start[i], row_start[i+1]);
  }

}


/*
   converts CSR format to CSC format, not in-place,
   if a == NULL, only pattern is reorganized.
   the size of matrix is n x m.
 */

void csr2csc(int n, int m, int nz, double *a, int *col_idx, int *row_start,
             double *csc_a, int *row_idx, int *col_start)
{
  int i, j, k, l;
  int *ptr;

  for (i=0; i<=m; i++) col_start[i] = 0;

  /* determine column lengths */
  for (i=0; i<nz; i++) col_start[col_idx[i]+1]++;

  for (i=0; i<m; i++) col_start[i+1] += col_start[i];


  /* go through the structure once more. Fill in output matrix. */

  for (i=0, ptr=row_start; i<n; i++, ptr++)
    for (j=*ptr; j<*(ptr+1); j++){
      k = col_idx[j];
      l = col_start[k]++;
      row_idx[l] = i;
      if (a) csc_a[l] = a[j];
    }

  /* shift back col_start */
  for (i=m; i>0; i--) col_start[i] = col_start[i-1];

  col_start[0] = 0;
}