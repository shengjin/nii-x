/////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#include "alloc.h"

extern int N_parm;
extern char *results_dir;




int max(int a, int b);
int min(int a, int b);

// uniform int between a and b
int i4_unif_ab(int a, int b);
// uniform int between 0 and a
int i4_unif_0a( int a );


// save the Beta values
int save_debug_Beta_Values(double *running_Beta_Values, int n_ranks);


// declaration of the mpi_batch function
int mpi_run_a_batch(MPI_Status status, int my_rank, int n_ranks, int root_rank, int rootsent_tag, int slavereturn_tag, double **transit_BetaParm_root, int n_iter_a_batch, unsigned i_save_begin, int nline_data, double *data_NlineNdim, double *ptr_sigma_prop, unsigned *ptr_i_accumul, unsigned *ptr_i_accumul_accept, double *logpost_allranks_root, double *running_Beta_Values, double **transit_BetaParmdx_root);


int mpi_static_sigma_stack(MPI_Status status, int my_rank, int n_ranks, int root_rank, int rootsent_tag, int slavereturn_tag, double **transit_BetaParm_root, int n_iter_a_stack, int n_iter_a_batch_base, int n_iter_a_batch_rand, unsigned i_save_begin, int nline_data, double *data_NlineNdim, double *ptr_sigma_prop, unsigned *ptr_i_accumul, unsigned *ptr_i_accumul_accept, double *logpost_allranks_root, double *running_Beta_Values, double **transit_BetaParmdx_root)
{    
    //
    int i_tmp = 0; 
    int i_next = 0; 
    //
    int n_iter_a_batch_adjust;
    int n_iter_a_batch = 0;
    //
    //
    while ( i_next < n_iter_a_stack )
    {
        /////////////////////////////////////////////
        //
        // Calc n_iter_a_batch, i_next
        if (my_rank == root_rank)
        {
            //
            n_iter_a_batch_adjust = i4_unif_ab(-n_iter_a_batch_rand, n_iter_a_batch_rand);
            n_iter_a_batch = n_iter_a_batch_base + n_iter_a_batch_adjust;
            //
            // new i_next
            i_next = i_tmp + n_iter_a_batch;
            if ( i_next > n_iter_a_stack )
            {
                n_iter_a_batch = n_iter_a_stack - i_tmp;
                i_next = n_iter_a_stack;
            }
            //
            // Tell slaves the i_next and n_iter_a_batch
            for(int i_rank = 0; i_rank < n_ranks; i_rank++)
            {
                if (i_rank != root_rank)
                {
                    //...tell a slave process i_next
                    MPI_Send(&i_next, 1 , MPI_INT, i_rank, rootsent_tag, MPI_COMM_WORLD);
                    //...tell a slave process how many iterations in a sampling model
                    MPI_Send(&n_iter_a_batch, 1 , MPI_INT, i_rank, rootsent_tag+1, MPI_COMM_WORLD);
                }
            }
        }
        else
        {
            // Receive the i_next
            MPI_Recv(&i_next, 1, MPI_INT, root_rank, rootsent_tag, MPI_COMM_WORLD, &status);
            // Receive the n_iter_a_batch
            MPI_Recv(&n_iter_a_batch, 1, MPI_INT, root_rank, rootsent_tag+1, MPI_COMM_WORLD, &status);
        }
        //
        // 
        // sync
        MPI_Barrier(MPI_COMM_WORLD);
        //
        //
        ////////////////////////////////////////////////////
        //
        // Run a batch
        mpi_run_a_batch(status, my_rank, n_ranks, root_rank, rootsent_tag, slavereturn_tag, transit_BetaParm_root, n_iter_a_batch, i_save_begin, nline_data, data_NlineNdim, ptr_sigma_prop, ptr_i_accumul, ptr_i_accumul_accept, logpost_allranks_root, running_Beta_Values, transit_BetaParmdx_root);
        //
        // sync since we need to pass logpost to the ranks in mpi_batch
        MPI_Barrier(MPI_COMM_WORLD);
        //
        //
        ///////////////////////////////////////////////
        //
        //
        // sync since we need to pass logpost to the ranks in mpi_batch
        MPI_Barrier(MPI_COMM_WORLD);
        //
        //
        // Update i_tmp
        i_tmp = i_next;
    }
    //
    //
    return 0;
}





//////////////////////////
// Find maximum between two numbers.
int max(int a, int b)
{
    return ( a > b ) ? a : b;
}


//////////////////////////
// Find minimum between two numbers.
int min(int a, int b)
{
    return ( a > b ) ? b : a;
}





int save_debug_stack_sequence(unsigned *ptr_i_accumul, int i_swap)
{
    FILE *out;
    //
    // set fnames
    char fname[100];

    snprintf(fname, sizeof fname, "%s%s%s", results_dir, "/", "swap_sequence.dat");
    //printf("%s\n", fname);
    //
    // open files
    if ((out = fopen(fname, "a")) == NULL)
    {
        fprintf(stderr, "Can't create output file!\n");
        exit(3);
    }
    //
    fprintf(out, "%u", *ptr_i_accumul);
    fprintf(out, "   ");
    fprintf(out, "%d", i_swap);
    fprintf(out, "\n");
    //
    // close files
    if (fclose(out) != 0)
        fprintf(stderr, "Error in closing file!\n");
    //
    return 0;
}
//
//
//


int save_debug_Beta_Values(double *Beta_Values, int n_ranks)
{
    FILE *out;
    //
    // set fnames
    char fname[100];

    snprintf(fname, sizeof fname, "%s%s%s", results_dir, "/", "Beta_Values_in_stack.dat");
    //printf("%s\n", fname);
    //
    // open files
    if ((out = fopen(fname, "a")) == NULL)
    {
        fprintf(stderr, "Can't create output file!\n");
        exit(3);
    }
    //
    //
    for(int i = 0; i < n_ranks; i++)                                                                                             
    {                                                                                                                            
        fprintf(out, "%lf ", Beta_Values[i]);                                                                         
    }                                                                                                                        
    fprintf(out, "\n");                                                                                                      
    //
    //
    // close files
    if (fclose(out) != 0)
        fprintf(stderr, "Error in closing file!\n");
    //
    return 0;
}



