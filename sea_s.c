#include <stdio.h>
#include <math.h>
#include <mpi.h>
#define N 100
#define pi 3.1415926
#define Niter 2000
#define esp 0.0001
#define epsilon (1./8)
void main(){

float u[N][N] = {0.0}, u_prev[N][N] = {0.};          // initialization
int ii, jj, iter, p = 0;                             // p is a variable to monitor the degree of convergence
double h = 1./(N-1), A, B;                           // h is step size from 0 to 1
double t1, t2;
A = h*h/(4*epsilon);
B = h/(8*epsilon);

t1 = MPI_Wtime();                                    // start time

/***********************************************************************************************************
Jacobi iteration, set p to monitor iteration , when it reaches 100% convergence, stop iteration and print u
***********************************************************************************************************/
 for(iter=1;iter<=Niter;iter++){

    for(ii=2;ii<=N-1;ii++){

        for(jj=2;jj<=N-1;jj++){

            u[ii-1][jj-1] = (u_prev[ii][jj-1] +
                             u_prev[ii-2][jj-1] +
                             u_prev[ii-1][jj] +
                             u_prev[ii-1][jj-2])/4 +
                             A*sin(pi*h*(jj-1)) +
                             B*(u_prev[ii][jj-1] -
                             u_prev[ii-2][jj-1]);

            if(fabs(u[ii-1][jj-1] - u_prev[ii-1][jj-1]) < esp) p++;
        }//jj
    }//i

    printf("In the iteration %d, %.2f percent matrix element is similaritiy to the previous ones in the error of 0.0001 \n",iter,100.*(4*N-4+p)/(N*N));
/********* when all element is similarity to previous iteration ones,end it and print *******/

    if(p == (N-2)*(N-2)){

        printf("Iteration : %d\n",iter);

        for(ii=1;ii<=N;ii++){

            for(jj=1;jj<=N;jj++){

                printf("%.2f\t",u[ii-1][jj-1]);
            }//jj

            printf("\n");
        }//ii
        printf("\n");

        break;

    }//if

    p = 0;                             // atfer each iteration, return p to 0
/********* when all element is similarity to previous iteration ones,end it and print *******/

/************************** give u to u_prev for next iteraion *****************************/
    for(ii=1;ii<=N;ii++){

        for(jj=1;jj<=N;jj++){

            u_prev[ii-1][jj-1] = u[ii-1][jj-1];
           // printf("%.4f\t",u[ii-1][jj-1]);

        }//jj
    }//ii
/************************** give u to u_prev for next iteraion *****************************/


 }//iter

t2 = MPI_Wtime();

printf("I am serial program, my time is %.6f \n",t2-t1);

}
