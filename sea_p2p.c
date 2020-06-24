#include <stdio.h>
#include <mpi.h>
#include <math.h>
#define N 100
#define pi 3.1415926
#define Niter 2000
#define epsilon (1./8)
#define esp 0.0001
void main(){
float u[N][N] = {0.0},u_prev[N][N] = {0.};
int ii,jj,iter;

int size,myid;
int local_start,local_finish;
int local_start_real,local_finish_real,block_size;
int local_p = 0,p = 0;
double local_t1,local_t2;
double t1,t2;

double h = 1./(N-1),A,B;
A = h*h/(4*epsilon);
B = h/(8*epsilon);

/****************************************************************************************************
1.Divides tasks.
2.Each process do its Jacobi iteration task, shares only the upper boundary to its upper process and
  the lower one to its lower one for the next iteration;
3.Each process get a local_p to monitor the numble of the same elements of the two consecutive
   iteration, each iteration will gathers all localP to p, when p is equal to (N-2)*(N-2),stop iteration 
  and gather all  local u to the master process which will print u then.
****************************************************************************************************/
MPI_Init(NULL,NULL);
MPI_Comm_size(MPI_COMM_WORLD,&size);
MPI_Comm_rank(MPI_COMM_WORLD,&myid);

block_size = N/size*N;
local_start_real = N/size*myid+1;
local_finish_real = N/size*(myid+1);
local_start = local_start_real;
local_finish = local_finish_real;
if(myid==0) local_start_real=2;
if(myid==size-1) local_finish_real = N-1;

MPI_Barrier(MPI_COMM_WORLD);
local_t1 = MPI_Wtime();

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
   100%( u==u_prev), stop the iteration, gather all local u
   to the master process whic will print u;
2. Allreduce makes sure all matrix elements have been 
   similarity to theoretical result;
3. p==(N-2)*(N-2),because the boundaries don't participate 
   in the iteration ;
4. Return p and local_p to 0 after each iteration.
***********************************************************/
     MPI_Allreduce(&local_p,&p,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
     if(myid==0)  printf("In the iteration %d, %.2f percent matrix element is similarity to the previous ones in the error of 0.001\n",iter,100.*(4*N-4+p)/(N*N));
   
     if(p==(N-2)*(N-2)) {
         MPI_Gather(&u[local_start-1][0],block_size,MPI_FLOAT,&u_prev[0][0],block_size,MPI_FLOAT,0,MPI_COMM_WORLD);
        
         break;
     }
     local_p=0;
     p=0;

/************************************************************
1. p2p share the upper data to the upper process,
   lower ones to lower one.
************************************************ **********/

     if(myid != 0) {
         MPI_Send(&u[local_start-1][1],N-2,MPI_FLOAT,myid-1,0,MPI_COMM_WORLD);
         MPI_Recv(&u_prev[local_start-2][1],N-2,MPI_FLOAT,myid-1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
     }
     if(myid != (size-1)) {
         MPI_Send(&u[local_finish-1][1],N-2,MPI_FLOAT,myid+1,1,MPI_COMM_WORLD);
         MPI_Recv(&u_prev[local_finish][1],N-2,MPI_FLOAT,myid+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
     }
/*************************************************************
1. share u to u_prev for next iteration
************************************************************/
     for(ii=local_start_real;ii<=local_finish_real;ii++){ 
        
         for(jj=2;jj<=N-1;jj++){
            
             u_prev[ii-1][jj-1] = u[ii-1][jj-1];
         }//jj
     }//ii


 }//iter

local_t2 = MPI_Wtime();
MPI_Reduce(&local_t1,&t1,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
MPI_Reduce(&local_t2,&t2,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

/*************************************************************
1.The master process print iteration result.
*************************************************************/
 if(myid==0) {

     printf("Iteration %d\n",iter);
     for(ii=1;ii<=N;ii++){

         for(jj=1;jj<=N;jj++){

             printf("%.2f\t",u_prev[ii-1][jj-1]);
         }//jj
         printf("\n");
     }//ii
    
     printf("\n");
     printf("I am paralle program mpi_p2p, my time = %.6f seconds\n",t2-t1);
 }

MPI_Finalize();

}
