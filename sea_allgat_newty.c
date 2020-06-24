#include <stdio.h>
#include <mpi.h>
#include <math.h>
#define N 100
#define pi 3.1415926
#define Niter 2000
#define epsilon (1./8)
#define esp 0.0001
void main(){
float u[N][N] = {0.0}, u_prev[N][N] = {0.};     //initializatin
int ii, jj, iter;
int np, myid;
int local_start, local_finish, local_start_real, local_finish_real, block_size;
int local_p = 0,p = 0;
double local_t1, local_t2, t1, t2;
double h = 1./(N-1),A,B;
A = h*h/(4*epsilon);
B = h/(8*epsilon);

/***************************************************************************************************************
1.Each process do its Jacobi iteration task, and share only its edge data to all for next iteration;
2.Each iteration will display p which reflects the degree of iteration instead of printing u,
  when all elements of u is similarity to u_prev, stop iteration and all process send their data
  to the master process which will print u then;  
***************************************************************************************************************/

MPI_Init(NULL,NULL);
MPI_Comm_size(MPI_COMM_WORLD,&np);
MPI_Comm_rank(MPI_COMM_WORLD,&myid);
MPI_Datatype newtype;                               // define a derived type

/***************************************************************************
1.Divide tasks.
 newtype: count=2 means the local start row and the local finish row
 blocklength=N,because each row has N element, stride is because each process 
 get N/np row ,each row has N elements, so the first row and last one have a
 stride of (N/np-1)*N.
***************************************************************************/
MPI_Type_vector(2,N,(N/np-1)*N,MPI_FLOAT,&newtype); 
MPI_Type_commit(&newtype);

block_size = N/np*N;
local_start_real = N/np*myid+1;
local_finish_real = N/np*(myid+1);
local_start = local_start_real;
local_finish = local_finish_real;
if(myid==0) local_start_real = 2;         // the master process doesn't change the boundary conditions
if(myid==np-1) local_finish_real = N-1;   // the same to the last one

MPI_Barrier(MPI_COMM_WORLD);
local_t1=MPI_Wtime();

/**************************************************************************
Jacobi iteration
**************************************************************************/
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
   
    if(p==(N-2)*(N-2)){    
        MPI_Gather(&u[local_start-1][0],block_size,MPI_FLOAT,&u_prev[0][0],block_size,MPI_FLOAT,0,MPI_COMM_WORLD);
        break;
    }//if

    local_p = 0;
    p = 0;

/**********************************************************
1. Share only the upper and lower boundaries to all process 
2. give u to u_prev for next iteration.
**********************************************************/

    MPI_Allgather(&u[local_start-1][0],1,newtype,&u_prev[0][0],1,newtype,MPI_COMM_WORLD);

    for(ii=local_start_real;ii<=local_finish_real;ii++){ 
        
        for(jj=2;jj<=N-1;jj++) u_prev[ii-1][jj-1] = u[ii-1][jj-1];
    }//ii
    
 }//iter


local_t2 = MPI_Wtime();
MPI_Reduce(&local_t1,&t1,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
MPI_Reduce(&local_t2,&t2,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

/*********************************************************
1. The master process prints the resurt.
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

MPI_Type_free(&newtype); 
MPI_Finalize();

}
