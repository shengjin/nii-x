
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <adolc/adolc.h>


extern int N_parm; 

// "logll_tempered" and "grad" are passed in as pointers and their values are updated inside the logll_beta_grad function.
extern "C" {
    int  logll_beta_grad(double *ptr_one_chain, int nline_data, double *data_NlineNdim, double beta_one, double *logll_tempered, double *grad)
    {
    // ptr_one_chain is the passive x (independant variables).
    // grad_dydx is the gradient.
    
    // tag_ID
    int tag = 0;
    trace_on(tag);

    // alloc new active x 
    adouble *x;
    x = new adouble[N_parm];
    //
    adouble y;

/*
    if (beta_one == 1.0)
    {
        printf("%.15f   %.15f\n", *ptr_one_chain, *(ptr_one_chain+1));
    }
*/

    x[0] <<= *ptr_one_chain;
    x[1] <<= *(ptr_one_chain+1);

    // test if adouble is needed?
    adouble ll;
    //ll =  -x[0]*x[0]*6 -x[1]*x[1]*8;
    ll = exp(-pow(x[0],2) - pow((9+4*pow(x[0],2) + 9*x[1]),2)) + 0.5 * exp(-8*pow(x[0],2)-8*pow((x[1]-2),2));
    //exp(-pow(a,2) - pow((9+4*pow(a,2) + 9*b),2)) + 0.5 * exp(-8*pow(a,2)-8*pow((b-2),2));
    //ll = exp(-pow(x[0],2) - pow((9+4*pow(x[0],2) + 9*x[1]),2)); // + 0.5 * exp(-8*pow(x[0],2)-8*pow((x[1]-2),2));
    //ll = exp(-pow(x[0],2) - pow((9+4*pow(x[0],2) + 9*x[1]),2)); // + 0.5 * exp(-8*pow(x[0],2)-8*pow((x[1]-2),2));
    //printf("0_logll, beta: %.15f %.15f", ll.value(), beta_one);
    
    y = ll*beta_one;

    y >>= *logll_tempered;
    //printf("1_logll: %.15f %.15f \n", y.value(), *logll_tempered);

    delete[] x;


    trace_off();
    
    gradient(tag, N_parm, ptr_one_chain, grad);


    // check derivertive 
    /*
    if (beta_one == 1.0)
    {
        FILE *fp = fopen("test.dat", "a");
        if (fp != NULL) {
            fprintf(fp, "x0dx0x1dx1  %.15f  %.15f %.15f  %.15f\n", *ptr_one_chain, grad[0], *(ptr_one_chain+1), grad[1]);
            fclose(fp);
        }
    }
    */

    return 0;
    //
    }
}

    
        
        
        
    
    






    






