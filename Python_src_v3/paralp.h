
int ratio_test1(double *dy, int *idy,int ndy, double *y, double *ybar, double mu);
void paralp(double *obj, double *mat, double *rhs, int *m0 , int *n0, 
            double *opt, int *status, double *lambda_min, double *rhs_bar, double *obj_bar);
void solver21(int m,int n,int nz,int *ia, int *ka, double *a,double *b, double *c);