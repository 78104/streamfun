#include <stdio.h>
#include <mpi.h>
#include <math.h>
#define N 100
#define pi 3.1415926
#define Niter 2000
#define epsilon (1./8)
#define esp 0.0001
void main(){

float u[N][N] = {0.0}, u_prev[N][N] = {0.};             // initialization
int ii, jj, iter;
int np, myid;                                           // for parallel mpi
int local_start, local_finish;                          // each process get an area 
int local_start_real, local_finish_real, block_size;    // for process 0 and the last one
int local_p = 0,p = 0;                                  // for the monitoration of convergence
double local_t1, local_t2, t1, t2;
double h = 1./(N-1), A, B;                              // step from 0 to 1
A = h*h/(4*epsilon);
B = h/(8*epsilon);

/************************************************************************************************
1.Each process does its Jacobi iteration task which need MPI_Allgather to share data , and set 
  local_p record to record the numble of the same element of two consecutive iterations,afer each
  iteration, gathered  to p used to monitor the degree of iteration and stop iteration when it 
  reaches (N-2)*N-2).
2.The master process print last u 
************************************************************************************************/

MPI_Init(NULL,NULL);
MPI_Comm_size(MPI_COMM_WORLD,&np);
MPI_Comm_rank(MPI_COMM_WORLD,&myid);

block_size = N/np*N;                          // assign caculation area
local_start_real = N/np*myid+1;
local_finish_real = N/np*(myid+1);
local_start = local_start_real;
local_finish = local_finish_real;
if(myid==0) local_start_real = 2;            // the first process doesn't caculate the first row   
if(myid==np-1) local_finish_real = N-1;      // the last one doesn't caculate the last row

MPI_Barrier(MPI_COMM_WORLD);
local_t1=MPI_Wtime();

 for(iter=1;iter<=Niter;iter++){

    for(ii=local_start_real;ii<=local_finish_real;ii++){
        
        for(jj=2;jj<=N-1;jj++){
            
            u[ii-1][jj-1] = (u_prev[ii][jj-1] + 
                            u_prev[ii-2][jj-1] +
                            u_prev[ii-1][jj] +  
                            u_prev[ii-1][jj-2])/4 +
                            A*sin(pi*h*(jj-1)) +
                            B*(u_prev[ii][jj-1] - u_prev[ii-2][jj-1]);
            
            if(fabs(u[ii-1][jj-1] - u_prev[ii-1][jj-1]) < esp)  local_p++; 
            
        }//jj
    }//ii

/*********************************************************** 
1. Display the degree of each iteration,when it reaches
   100%( u==u_prev), stop the iteration;
2. Allreduce makes sure all matrix elements have been 
   similarity to theoretical result;
3. p==(N-2)*(N-2),because the boundaries don't participate 
   in the iteration ;
4. Return p and local_p to 0 after each iteration.
***********************************************************/
    MPI_Allreduce(&local_p,&p,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
   
    if(myid==0)  printf("In the iteration %d, %.2f percent matrix element is similarity to the previous ones in the error of 0.001\n",iter,100.*(4*N-4+p)/(N*N));
   
    if(p==(N-2)*(N-2))  break;
                              
    local_p=0;      
    p=0;                     


    MPI_Allgather(&u[local_start-1][0],block_size,MPI_FLOAT,&u_prev[0][0],block_size,MPI_FLOAT,MPI_COMM_WORLD);
    //share all data to all process for next iteration

 }//iter

local_t2=MPI_Wtime();
MPI_Reduce(&local_t1,&t1,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
MPI_Reduce(&local_t2,&t2,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

/*********************************************************
 The master proccess prints iteration result.
 *********************************************************/
if(myid==0) {

    printf("Iteration %d\n",iter);
    for(ii=1;ii<=N;ii++){

        for(jj=1;jj<=N;jj++){

            printf("%.2f\t",u_prev[ii-1][jj-1]);
        }//jj
        printf("\n");
    }//ii
    
    printf("\n");
    printf("I am paralle program mpi, my time = %.6f seconds\n",t2-t1);
 }

MPI_Finalize();

}
