
/**
 * Parallel CUDA algorithm for border tracking Víctor Manuel García Mollá (vmgarcia@dsic.upv.es)
 *
 */

#include <math.h>
#include <iostream>
#include <stdio.h>
typedef unsigned char uint8_t; ///< uint8_t type definition. Contains values in range [0, 255], using 8 bits (1 uint8_t)

#define CUDA_SAFE_CALL( call, routine ) { \
 cudaError_t err = call; \
 if( cudaSuccess != err ) { \
   fprintf(stderr,"CUDA: error %d occurred in %s routine. Exiting...\n", err, routine); \
   exit(err); \
 } \
}

//definition of variables for openCV
//#define OPENCV

#ifdef OPENCV
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;
using namespace std;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
#endif

// data of test image


#define	M			1232  //dimensions of image
#define	N			1028
#define	FILENAME		"Frame_08.bin"
#define	N_BLOCKS_ROWS	32	// 4
#define	N_BLOCKS_COLS	32   // 4
//Dimensions of augmented image
#define	Mg		1248 //  first multiple of N_BLOCKS_ROWS greater than 1232 (M)
#define	Ng	    1056  //  first multiple of N_BLOCKS_COLS greater than 1028 (N)
#define FILENAMEBMP     "Frame_08.bmp" //version for Opencv

//auxiliar macros; the variables for images ended in g are augmented matrices, so that their size is multiple of the block size

//d_A is the original image; and h_A is the original size output image,
#define	h_A(i,j)		h_A[ (i) + ((j)*(M)) ]
#define	h_Ag(i,j)		h_Ag[ (i) + ((j)*(Mg)) ]
#define	h_Asal(i,j) 		h_Asal[ (i) + ((j)*(M)) ]
#define	d_A(i,j) 		d_A[ (i) + ((j)*(Mg)) ]  //#define	d_A(i,j) 		d_A[ (i) + ((j)*(M)) ]
#define	d_Ag(i,j,ldg) 		d_Ag[ (i) + ((j)*(Mg)) ]
#define	d_Asal(i,j) 		d_Asal[ (i) + ((j)*(M)) ]
#define	d_is_bord(i,j) 		d_is_bord[ (i) + ((j)*(Mg)) ]

#define indice_despl_x(i,j)	indice_despl_x[ (i) + ((j)*(8)) ]


// Maximum number of threads per block of K20Xm: 1024 = 32*32
#define THREADS_PER_BLOCK_X 32		//Para preproceso
#define THREADS_PER_BLOCK_Y 32	//Para preproceso


#define MAX_N_BORDS		500 //Max number of borders by rectangle of image, in this version



struct coord {
    int i;
    int j;
};

// structure to store coordinate of each point in a contour
struct VecCont {
    coord act;  // Coordenadas del punto actual
    coord sig;  // Coordenadas del punto siguiente
    coord ant;  // Coordenadas del punto anterio
    int next;   // Indice del punto siguiente dentro de un vector de VecCont
};

typedef enum { OPEN_CONTOUR, CLOSED_CONTOUR, COVERED_CONTOUR } con;

// structure of positions of ech contour
struct IndCont {
    int ini;    // Index of initial point of in a vector of veccont
    int fin;    // Index of final point of in a vector of veccont
    con sts;    // contour state (open, closed, or covered)

};


/**
 * routines for cpu timing
 */
clock_t cpu_startTime, cpu_endTime;
double cpu_ElapsedTime;
double cpu_ElapsedTimes[3][1000];

void startCPUTimer(void) {
    cpu_startTime = clock();
}

void stopCPUTimer(const char* text) {
    cpu_endTime = clock();
    cpu_ElapsedTime = ( (cpu_endTime - cpu_startTime) / (double) ( CLOCKS_PER_SEC / 1000 ) );
    printf("%s: %f ms (CPU)\n", text, cpu_ElapsedTime);
}

void saveCPUTimer(int step, int run) {
    cpu_endTime = clock();
    cpu_ElapsedTimes[step][run] = ( (cpu_endTime - cpu_startTime) / (double) ( CLOCKS_PER_SEC / 1000 ) );
}

void averageCPUTimer(int step, int runs, const char* text) {
    cpu_ElapsedTime = 0;
    // If there are more than one execution, discard the first one
    for( int run = (runs > 1 ? 1 : 0); run < runs; run++ ) {
        cpu_ElapsedTime += cpu_ElapsedTimes[step][run];
    }
    if ( runs == 1) {
        printf("%s: %f ms (CPU)\n", text, cpu_ElapsedTime );
    } else {
        printf("%s: %f ms (CPU average of %d executions)\n", text, cpu_ElapsedTime / (runs - 1), runs - 1 );
    }
}


/**
 * routines for GPU timing
 */
cudaEvent_t start, stop;
float gpu_ElapsedTime;
float gpu_ElapsedTimes[8][1000];

void startCudaTimer(void) {
    cudaEventRecord(start, 0);
}

void stopCudaTimer(const char* text) {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_ElapsedTime, start, stop);
    printf("%s: %f ms (GPU)\n", text, gpu_ElapsedTime);
}

void saveCudaTimer(int step, int run) {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_ElapsedTimes[step][run], start, stop);
}

void averageCudaTimer(int step, int runs, const char* text) {
    float gpu_ElapsedTime = 0;
    // If there are more than one execution, discard the first one
    for( int run = (runs > 1 ? 1 : 0); run < runs; run++ ) {
        gpu_ElapsedTime += gpu_ElapsedTimes[step][run];
    }
    if ( runs == 1) {
        printf("%s: %f ms (GPU)\n", text, gpu_ElapsedTime );
    } else {
        printf("%s: %f ms (GPU average of %d executions)\n", text, gpu_ElapsedTime / (runs - 1), runs - 1 );
    }
}





/**
 * Kernel CUDA: detection of contour points
 */
__global__ void preprocessing_gpu(

    uint8_t *d_A,   //input image
    uint8_t *d_is_bord  //output binary array, indicating whether pixel (i,j) is contour (d_is_bord(i,j)=1) or not (d_is_bord(i,j)=0)

) {
    __shared__ uint8_t arr_sh[(THREADS_PER_BLOCK_X + 2)*(THREADS_PER_BLOCK_Y + 2)];
//	buffer of shared memory
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int pos = i + 1 + Mg * (j + 1);
    int pos_local = threadIdx.x + 1 + (threadIdx.y + 1)*(blockDim.x + 2);
    int ilocal = threadIdx.x + 1;
    int jlocal = threadIdx.y + 1;
    int lda_shared = blockDim.x + 2;
    int cond, pos_zero = 0;

    //load piece of image on shared memory buffer
    if ((i < M - 1) && (j < N - 1)) {
        arr_sh[pos_local] = d_A[pos];
        if (threadIdx.x == 0)
            arr_sh[ilocal - 1 + (jlocal)*lda_shared] = d_A[i + Mg * (j + 1)];
        if (threadIdx.x == blockDim.x - 1)
            arr_sh[ilocal + 1 + (jlocal)*lda_shared] = d_A[i + 2 + Mg * (j + 1)];
        if (threadIdx.y == 0)
            arr_sh[ilocal + (jlocal - 1)*(lda_shared)] = d_A[i + 1 + Mg * (j)];
        if (threadIdx.y == blockDim.y - 1)
            arr_sh[ilocal + (jlocal + 1)*(lda_shared)] = d_A[i + 1 + Mg * (j + 2)];
        if ((threadIdx.x == 0) && (threadIdx.y == 0))
            arr_sh[ilocal - 1 + (jlocal - 1)*(lda_shared)] = d_A[i + Mg * (j)];
        if ((threadIdx.x == 0) && (threadIdx.y == blockDim.y - 1))
            arr_sh[ilocal - 1 + (jlocal + 1)*(lda_shared)] = d_A[i + Mg * (j + 2)];
        if ((threadIdx.x == blockDim.x - 1) && (threadIdx.y == blockDim.y - 1))
            arr_sh[ilocal + 1 + (jlocal + 1)*(lda_shared)] = d_A[i + 2 + Mg * (j + 2)];
        if ((threadIdx.x == blockDim.x - 1) && (threadIdx.y == 0))
            arr_sh[ilocal + 1 + (jlocal - 1)*(lda_shared)] = d_A[i + 2 + Mg * (j)];

    }
    __syncthreads();

    if ((i < M - 1) && (j < N - 1))
        //determine whether pixel(i,j) is a contour pixel
    {
        pos_zero = (arr_sh[ilocal - 1 + (jlocal)*(lda_shared)] == 0) * 2;
        cond = (pos_zero == 0)*(arr_sh[ilocal + (jlocal + 1)*(lda_shared)] == 0);
        pos_zero = 4 * cond + (1 - cond)*pos_zero;
        cond = (pos_zero == 0)*(arr_sh[ilocal + 1 + (jlocal)*(lda_shared)] == 0);
        pos_zero = 6 * cond + (1 - cond)*pos_zero;
        cond = (pos_zero == 0)*(arr_sh[ilocal + (jlocal - 1)*(lda_shared)] == 0);
        pos_zero = 8 * cond + (1 - cond)*pos_zero;
        d_is_bord[pos] = (arr_sh[pos_local] > 0) && (pos_zero > 0);


    }

}

static __constant__ int iouts[3][3] = {
    {0, -1, -1},
    {1, 0, -1},
    {1, 1, 0}
};

static __constant__ int poss[3][3] = {
    {2, 1, 3},
    {1, 0, 1},
    {7, 1, 5}
};
__device__ void  clockwise_2(int difi, int difj, int *iout, int *jout, int* pos)
// function for obtaining next pixel rotating clockwise
//difi, dif j give relative position of actual pixel relative to center pixel
// example, if center pixel is (i,j) and difi=1, dif j=0 , present pixel is
//i+1, j+0.
//pos =2 =>(0, -1); pos=3 =>(-1, 0), pos =5 => 0,1, pos=7 => 1,0
{
    *iout = iouts[difi+1][-difj+1];
    *jout = iouts[-difi+1][-difj+1];
    *pos = poss[difj+1][-difi+1];
}

__device__ void  counterclock_2(int difi, int difj, int *iout, int *jout, int* pos)
// function for obtaining next pixel rotating counterclockwise
//difi, dif j give relative position of actual pixel relative to center pixel
// example, if center pixel is (i,j) and difi=1, dif j=0 , present pixel is
//i+1, j+0.
//pos =2 =>0, -1; pos=3 =>-1, 0, pos =5 => 0,1, pos=7 => 1,0
{
    *iout = iouts[difi+1][difj+1];
    *jout = iouts[difi+1][-difj+1];
    *pos = poss[difi+1][difj+1];
}


__device__ void track_fw_bkw(
    int *i_vec_conts_ini, //initial point of d_vec_conts vector where present contour is stored
    uint8_t *d_A,        //image
    uint8_t *d_is_bord,  //is_bord array
    int* d_numconts,  //index of contour being tracked
    VecCont* d_vec_conts, //vector of contour points
    IndCont* d_ind_conts,  //vector of contours
    int i_ind_conts,  //number of contour, already increased
    int i_ini,       //boundaries of rectangle i_ini,i_fin, j_ini, j_fin
    int j_ini,
    int i_fin,
    int j_fin,
    coord c_ini_ant,   //triad for start of tracking
    coord c_ini_act,
    coord c_ini_sig
) {
    coord coord_sig, coord_act, coord_ant;
    int dif_i, dif_j, itcount, i_vec_conts, iaux, val;
    int	found, jaux, pos;
    i_vec_conts = *i_vec_conts_ini;

    coord_sig = c_ini_sig;
    coord_act = c_ini_act;
    //set first point of border
    d_vec_conts[i_vec_conts].act = c_ini_act;
    d_vec_conts[i_vec_conts].ant = c_ini_ant;
    d_vec_conts[i_vec_conts].sig = c_ini_sig;
    d_ind_conts[i_ind_conts].ini = i_vec_conts;
    d_ind_conts[i_ind_conts].fin = i_vec_conts;
    d_ind_conts[i_ind_conts].sts = OPEN_CONTOUR;

    int end_track_forward = 0;
    //start tracking forward, checking if we are leaving the rectangle
    if ((coord_sig.i < i_ini) || (coord_sig.i > i_fin) || (coord_sig.j < j_ini) || (coord_sig.j > j_fin)) {
        d_vec_conts[i_vec_conts].next = 0; //leaving the rectangle

        end_track_forward = 2;
    } else {
        while (end_track_forward == 0) {
            //if we do not leave, add new point to contour
            i_vec_conts++;
            d_vec_conts[i_vec_conts].act = coord_sig;
            d_vec_conts[i_vec_conts].ant = coord_act;
            d_vec_conts[i_vec_conts - 1].next = i_vec_conts;
            coord_ant = coord_act;
            coord_act = coord_sig;


            dif_i = coord_ant.i - coord_act.i;
            dif_j = coord_ant.j - coord_act.j;
            found = 0;
            itcount = 0;
            val = d_A(coord_act.i, coord_act.j);
            //next while look for next "one" after (iant,jant),
            //and stores in  "val" the zeros found,  for updating pixel value, as in algorithm 3 in paper
            while ((found == 0) && (itcount <= 8)) {
                counterclock_2(dif_i, dif_j, &iaux, &jaux, &pos);
                if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0) {
                    val = val * pos;
                    dif_i = iaux;
                    dif_j = jaux;
                    itcount++;
                } else {
                    found = 1;
                    coord_sig.i = coord_act.i + iaux;
                    coord_sig.j = coord_act.j + jaux;
                }
            }
            if ((coord_sig.i < i_ini) || (coord_sig.i > i_fin) || (coord_sig.j < j_ini) || (coord_sig.j > j_fin)) {
                //if leaving rectangle, finish this piece of contour
                end_track_forward = 2;
                d_vec_conts[i_vec_conts].sig = coord_sig;
                d_ind_conts[i_ind_conts].fin = i_vec_conts;
                d_A(coord_act.i, coord_act.j) = val;
            }

            else {
                d_vec_conts[i_vec_conts].sig = coord_sig;

                d_A(coord_act.i, coord_act.j) = val;
            }
            //If contour "closes":
            if ((coord_sig.i == c_ini_act.i) && (coord_sig.j == c_ini_act.j) && (coord_act.i == c_ini_ant.i) && (coord_act.j == c_ini_ant.j)) {
                end_track_forward = 1;

                d_ind_conts[i_ind_conts].sts = CLOSED_CONTOUR;
                d_ind_conts[i_ind_conts].fin = i_vec_conts;
                d_vec_conts[i_vec_conts].next = (*i_vec_conts_ini);
                d_vec_conts[i_vec_conts].sig = c_ini_act;
                d_vec_conts[(*i_vec_conts_ini)].ant = coord_act;
            }
        }
    }
    if (end_track_forward == 2) {
        //the contour went out of rectangle; we go back to beginning and track backwards; very similar, but backwards
        coord_ant = c_ini_ant;
        coord_act = c_ini_act;
        coord_sig = c_ini_sig;
        int anterior = d_ind_conts[i_ind_conts].ini;
        if ((coord_ant.i < i_ini) || (coord_ant.i > i_fin) || (coord_ant.j < j_ini) || (coord_ant.j > j_fin))

            d_vec_conts[(*i_vec_conts_ini)].ant = coord_ant;
        else {
            int end_track_backward = 0;
            while (end_track_backward == 0) {

                i_vec_conts++;
                d_vec_conts[i_vec_conts].act = coord_ant;
                d_vec_conts[i_vec_conts].sig = coord_act;
                d_vec_conts[i_vec_conts].next = anterior;
                anterior = i_vec_conts;
                coord_sig = coord_act;
                coord_act = coord_ant;
                dif_i = coord_sig.i - coord_act.i;
                dif_j = coord_sig.j - coord_act.j;
                found = 0;
                itcount = 0;
                val = d_A(coord_act.i, coord_act.j);
                while (found == 0 && itcount <= 8) { //rotating, looking for next "backward" pixel
                    clockwise_2(dif_i, dif_j, &iaux, &jaux, &pos);
                    if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0) {
                        val = val * pos;
                        dif_i = iaux;
                        dif_j = jaux;
                        itcount++;
                    } else {
                        found = 1;
                        coord_ant.i = coord_act.i + iaux;
                        coord_ant.j = coord_act.j + jaux;
                    }
                }
                if ((coord_ant.i < i_ini) || (coord_ant.i > i_fin) || (coord_ant.j < j_ini) || (coord_ant.j > j_fin)) {
                    end_track_backward = 2;
                    d_vec_conts[i_vec_conts].ant = coord_ant;
                    d_ind_conts[i_ind_conts].ini = i_vec_conts;
                    d_A(coord_act.i, coord_act.j) = val;
                } else {
                    d_vec_conts[i_vec_conts].ant = coord_ant;
                    d_ind_conts[i_ind_conts].ini = i_vec_conts;
                    d_A(coord_act.i, coord_act.j) = val;
                }



            }
        }

    }
    (*i_vec_conts_ini) = i_vec_conts;
}
/**auxiliar kernels for parallel tracking */

__device__ void  clockwise_o(int difi, int difj, int *iout, int *jout) {
    *iout = iouts[difi+1][-difj+1];
    *jout = iouts[-difi+1][-difj+1];
}

__device__ void rotate_ini(uint8_t *d_A, coord *coord_ant, coord *coord_sig, int *found, int *val, int *pos_ult_cero, coord coord_act) {
    //first rotation around a pixel, looking for a first not covered triad to start tracking it;
    //line 5 , Algorithm 1 of paper
    int dif_i = 0, dif_j = -1;
    int itcount, iaux, jaux, iaux2, jaux2;
    if (d_A(coord_act.i, coord_act.j - 1) != 0) {
        *found = 0;
        itcount = 0;
        while (*found == 0 && itcount <= 4) {
            clockwise_o(dif_i, dif_j, &iaux2, &jaux2);
            clockwise_o(iaux2, jaux2, &iaux, &jaux);
            if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0) {
                *found = 1;
                dif_i = iaux;
                dif_j = jaux;
            } else {
                dif_i = iaux;
                dif_j = jaux;
                itcount = itcount + 1;
            }
        }
    }
    *found = 0;
    itcount = 0;
    while (*found == 0 && itcount <= 8) { //lookfor pixel greater than zero, clockwise
        clockwise_o(dif_i, dif_j, &iaux, &jaux);
        if (d_A(coord_act.i + iaux, coord_act.j + jaux) != 0) {
            *found = 1;
            (*coord_ant).i = coord_act.i + iaux;
            (*coord_ant).j = coord_act.j + jaux;
        } else {
            dif_i = iaux;
            dif_j = jaux;
            itcount = itcount + 1;
        }
    }
    if (*found == 0) { //NO neighbor pixel greater than zero
        (*coord_ant).i = 0;
        (*coord_ant).j = 0;
        (*coord_sig).i = 0;
        (*coord_sig).j = 0;
        *val = d_A(coord_act.i, coord_act.j);
        *pos_ult_cero = 2;
    } else {
        *found = 0;
        itcount = 0;
        dif_i = iaux;
        dif_j = jaux;
        *val = 1;
        int pos = 0;
        while (*found == 0 && itcount <= 8) { //starting from former pixel, look for nex pixel counterclockwise
            counterclock_2(dif_i, dif_j, &iaux, &jaux, &pos);
            if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0) {
                *val = (*val)*pos; //storing in "val" the visited zeros
                if (pos > 1)
                    *pos_ult_cero = pos;
            }

            if (d_A(coord_act.i + iaux, coord_act.j + jaux) != 0) {
                *found = 1;
                (*coord_sig).i = coord_act.i + iaux;
                (*coord_sig).j = coord_act.j + jaux;
            } else {
                dif_i = iaux;
                dif_j = jaux;
                itcount = itcount + 1;
            }
        }
    }
}

__device__ void rotate_later(
    uint8_t *d_A, coord *coord_ant, coord *coord_sig, int *fin, int *val, int *pos_ult_cero, coord coord_act) {
    //next rotations around a pixel, looking for other not covered triads;
    //line 13 , Algorithm 1 of paper
    int dif_i = (*coord_ant).i - coord_act.i;
    int dif_j = (*coord_ant).j - coord_act.j;
    int itcount, found, pos, iaux, jaux;
    *coord_sig = *coord_ant;
    found = 0;
    itcount = 0;
    *fin = 0;
    *val = 1;
    while (found == 0 && itcount <= 8) { //look for zero pixel clockwise, starting from iant
        if (dif_i*dif_j == 0) { //one of the four  "cross " positions
            clockwise_2(dif_i, dif_j, &dif_i, &dif_j, &pos);
            if (d_A(coord_act.i + dif_i, coord_act.j + dif_j) != 0) {
                (*coord_sig).i = coord_act.i + dif_i;
                (*coord_sig).j = coord_act.j + dif_j;
            }
            clockwise_2(dif_i, dif_j, &iaux, &jaux, &pos);
        } else { //one of the four  "corner " positions
            clockwise_2(dif_i, dif_j, &iaux, &jaux, &pos);
        }
        if (d_A(coord_act.i + iaux, coord_act.j + jaux) != 0) {
            (*coord_sig).i = coord_act.i + iaux;
            (*coord_sig).j = coord_act.j + jaux;
        }
        if ((iaux == 0) && (jaux == -1)) {
            found = 1;
            *fin = 1;
        } else if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0) {
            found = 1;
            // iant=i+iaux;
            // jant=j+jaux;
        } else {
            dif_i = iaux;
            dif_j = jaux;
            itcount = itcount + 1;
        }

    }

    *pos_ult_cero = pos;
    *val = (*val)*pos;
    dif_i = iaux;
    dif_j = jaux;
    if ((*fin == 0) && (found == 1)) {
        found = 0;
        itcount = 0;
        while (found == 0 && itcount <= 8) {
            clockwise_2(dif_i, dif_j, &iaux, &jaux, &pos);
            if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0) {
                *val = (*val)*pos; //the visited "zeros" are accumulated in "pos"
                if (pos > 1) {
                    *pos_ult_cero = pos;
                }
            }
            if ((iaux == 0) && (jaux == -1)) {
                found = 1;
                (*coord_ant).i = coord_act.i + iaux;
                (*coord_ant).j = coord_act.j + jaux;
                if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0)
                    *fin = 1;
                else if (d_A(coord_act.i + iaux, coord_act.j + jaux) != 0)
                    *fin = 2;
            } else if (d_A(coord_act.i + iaux, coord_act.j + jaux) != 0) {
                found = 1;
                (*coord_ant).i = coord_act.i + iaux;
                (*coord_ant).j = coord_act.j + jaux;
            } else {
                dif_i = iaux;
                dif_j = jaux;
                itcount = itcount + 1;
            }
        }
    }
}







/**
 * Kernel CUDA: parallel_tracking
* kernel to be called using a single thread per block.
* Each thread (block) tracks the contours of its rectangle and stores them.
*Each thread visits all the pixels of its rectangle, if a contour pixel is found then it is tracked (using track_fw_bkw)
*and stored
*The "closed" contours (do not leave the rectangle) are directly stored in the global structure d_ind_conts_glob,
*using mutual exclusion to avoid data races
 *
 */
__global__ void parallel_tracking(
    uint8_t *d_A,
    uint8_t *d_is_bord,
    int* d_numconts,
    VecCont* d_vec_conts,
    IndCont* d_ind_conts,
    IndCont* d_ind_conts_glob,
    int* d_numconts_glob) {
    //a thread per block

    int ib = blockIdx.x; //thread index
    int jb = blockIdx.y;
    int mb = gridDim.x;   //number of blocks
    int nb = gridDim.y;
    int numfbl = Mg / mb;  //dimensions of rectangles
    int numcbl = Ng / nb;

    // limits of rectangle
    int i_ini = (ib == 0) ? 1 : (ib * numfbl);
    int j_ini = (jb == 0) ? 1 : (jb * numcbl);
    int i_fin = (ib == mb - 1) ? (M - 1) : ((ib + 1) * numfbl - 1);
    int j_fin = (jb == nb - 1) ? (N - 1) : ((jb + 1) * numcbl - 1);

    // block index (from 0 to mfb*nfb-1)
    int indicebl = ib + mb * jb;

    int i_vec_conts = (jb)*numcbl*Mg * 2 - 1 + 2 * numfbl*numcbl*(ib);
    //The *2 is because the vector of points has size 2*Mg*Ng

    // initial position - 1 of the contours of present block in the structure d_ind_conts
    int i_ind_conts = indicebl * MAX_N_BORDS - 1;

    coord coord_act, coord_ant, coord_sig;
    int found, val, pos_ult_cero;
    int borders_thispoint_tracked;
//main double loop to inspect all the points in the rectangle
    for (coord_act.j = j_ini; (coord_act.j <= j_fin); coord_act.j++) {
        for (coord_act.i = i_ini; (coord_act.i <= i_fin); coord_act.i++) {
            if (d_is_bord(coord_act.i, coord_act.j) > 0) { //only the contour points are processed
                rotate_ini(d_A, &coord_ant, &coord_sig, &found, &val, &pos_ult_cero, coord_act); //find first triad
                if (found != 0) {
                    //a triad has been found, not tracked yet
                    if ((d_A(coord_act.i, coord_act.j) == 1) || (d_A(coord_act.i, coord_act.j) % pos_ult_cero) != 0) {
                        d_A(coord_act.i, coord_act.j) *= val;
                        i_ind_conts++; //new contour
                        i_vec_conts++;
                        d_numconts[indicebl]++;
                        //d_vec_conts[i_vec_conts].act = coord_act;
                        track_fw_bkw(&i_vec_conts, d_A, d_is_bord, d_numconts, d_vec_conts, d_ind_conts, i_ind_conts, i_ini, j_ini, i_fin, j_fin, coord_ant, coord_act, coord_sig);
                        if (d_ind_conts[i_ind_conts].sts==CLOSED_CONTOUR) {
                            int contg=atomicAdd(d_numconts_glob,1);  //if closed contour, add to global structure
                            d_ind_conts_glob[contg+1]=d_ind_conts[i_ind_conts];
                            i_ind_conts--;
                            d_numconts[indicebl]--;
                        }
                        borders_thispoint_tracked = 0;
                        while (borders_thispoint_tracked == 0) {
                            rotate_later(d_A, &coord_ant, &coord_sig, &borders_thispoint_tracked, &val, &pos_ult_cero, coord_act); //search for new triads not tracked
                            if ((borders_thispoint_tracked != 1) && (d_A(coord_act.i, coord_act.j) % pos_ult_cero != 0)) {
                                d_A(coord_act.i, coord_act.j) *= val;
                                i_vec_conts++;
                                i_ind_conts++;
                                d_numconts[indicebl]++;
                                track_fw_bkw(&i_vec_conts, d_A,  d_is_bord, d_numconts, d_vec_conts, d_ind_conts, i_ind_conts, i_ini, j_ini, i_fin, j_fin, coord_ant, coord_act, coord_sig);
                                if (d_ind_conts[i_ind_conts].sts==CLOSED_CONTOUR) {
                                    int contg=atomicAdd(d_numconts_glob,1);
                                    d_ind_conts_glob[contg+1]=d_ind_conts[i_ind_conts];
                                    i_ind_conts--;
                                    d_numconts[indicebl]--;
                                }
                                if (borders_thispoint_tracked == 2) { //especial case
                                    borders_thispoint_tracked = 0;
                                }

                            }
                        }
                    } else if (d_A(coord_act.i, coord_act.j) > 1) {
                        borders_thispoint_tracked = 0;
                        while (borders_thispoint_tracked == 0) {
                            rotate_later(d_A, &coord_ant, &coord_sig, &borders_thispoint_tracked, &val, &pos_ult_cero, coord_act);
                            if ((borders_thispoint_tracked != 1) && (d_A(coord_act.i, coord_act.j) % pos_ult_cero != 0))

                            {
                                d_A(coord_act.i, coord_act.j) *= val;
                                i_vec_conts++;
                                i_ind_conts++;
                                d_numconts[indicebl]++;
                                track_fw_bkw(&i_vec_conts, d_A,  d_is_bord, d_numconts, d_vec_conts, d_ind_conts, i_ind_conts, i_ini, j_ini, i_fin, j_fin, coord_ant, coord_act, coord_sig);
                                if (d_ind_conts[i_ind_conts].sts==CLOSED_CONTOUR) {
                                    int contg=atomicAdd(d_numconts_glob,1);
                                    d_ind_conts_glob[contg+1]=d_ind_conts[i_ind_conts];
                                    i_ind_conts--;
                                    d_numconts[indicebl]--;
                                }
                            }
                        }
                    }
                } else if ((d_is_bord(coord_act.i, coord_act.j) > 0) && (found == 0)) { //contour of single point
                    i_vec_conts++;
                    //i_ind_conts++;
                    int contg=atomicAdd(d_numconts_glob,1);

                    coord_ant.i = coord_act.i;
                    coord_ant.j = coord_act.j;
                    coord_sig.i = coord_act.i;
                    coord_sig.j = coord_act.j;
                    d_ind_conts_glob[contg+1].sts = CLOSED_CONTOUR;
                    d_ind_conts_glob[contg+1].fin = i_vec_conts;
                    d_ind_conts_glob[contg+1].ini = i_vec_conts;
                    d_vec_conts[i_vec_conts].next = i_vec_conts;
                    d_vec_conts[i_vec_conts].sig = coord_sig;
                    d_vec_conts[i_vec_conts].act = coord_act;
                    d_vec_conts[i_vec_conts].ant = coord_ant;

                }
            }
        }







    }
}






/**
 * Kernel CUDA: Vertical_connection of borders . The closed borders are added to the global sructure
 *   Para cada bloque, creación de la lista de
 */
__global__ void vertical_connection(

    int num_max_conts,
    int     *d_numconts,
    VecCont *d_vec_conts,
    IndCont *d_ind_conts,
    int     *d_numconts_out,
    IndCont *d_ind_conts_out,
    int *marked,
    int numfbl,
    int numcbl,
    IndCont* d_ind_conts_glob,
    int* d_numconts_glob
) {

    int ib = blockIdx.x;
    int jb = blockIdx.y;
    int mb = 2*gridDim.x;
    int nb = gridDim.y;
    int i_con_ini, j_con_ini,i_antes, j_antes, i_salida, nump1,p2, jfc;
    int i_next_in, j_next_in, i_next_out,j_next_out, pf_bueno;
    int connected_border, ind_c_fuera;

    int ibloque, contorno, ind_b_fuera;
    int ibloque_arriba=ib*2+mb*(jb);
    int ibloque_sal_arriba=ibloque_arriba/2;
    //int ibloque_sal_arriba=ib+mb*(jb);
    int ibloque_abajo=(ib*2)+1+mb*(jb);
    int indice_ini=num_max_conts*(ibloque_arriba);
    int indice_ini_abajo=num_max_conts*(ibloque_abajo);
    int numcslocal=-1;
    int i_ind_conts=numcslocal+indice_ini;
    for (int cont=0; cont< d_numconts[ibloque_arriba]; cont++) {
        ibloque=ibloque_arriba;
        int indcactual=cont+indice_ini;

        if (marked[indcactual]==0) {
            numcslocal++;
            i_ind_conts++;

            d_ind_conts_out[i_ind_conts]=d_ind_conts[indcactual];
            int nump1=	d_ind_conts[indcactual].ini;
            i_con_ini=d_vec_conts[nump1].act.i;
            j_con_ini=d_vec_conts[nump1].act.j;
            i_antes=d_vec_conts[nump1].ant.i;
            j_antes=d_vec_conts[nump1].ant.j;
            int fin=0;
            contorno=cont;
            i_salida=ibloque_abajo;
            while(fin==0) {
                int indcaux=contorno+num_max_conts*(ibloque);
                //  final point of contour
                int p2 = d_ind_conts[indcaux].fin;

                // Coords of final point of contour, in present block
                int i_con_dentro = d_vec_conts[p2].act.i;
                int j_con_dentro = d_vec_conts[p2].act.j;
                // next point of contour, out of present block
                int i_con_fuera = d_vec_conts[p2].sig.i;
                int j_con_fuera = d_vec_conts[p2].sig.j;
                // Index of block where next pixel is
                int ibl_siguiente =  i_con_fuera / numfbl ;
                if ( ibl_siguiente > mb-1 )
                    ibl_siguiente = mb-1;
                int jbl_siguiente = j_con_fuera / numcbl ;
                if ( jbl_siguiente > nb-1 )
                    jbl_siguiente = nb-1;

                ind_b_fuera=ibl_siguiente+mb*(jbl_siguiente);
                if (ind_b_fuera!=i_salida) {
                    fin =1;
                    d_ind_conts_out[i_ind_conts].ini=nump1;
                    d_ind_conts_out[i_ind_conts].fin=p2;
                    d_ind_conts_out[i_ind_conts].sts=OPEN_CONTOUR;
                    marked[indcactual]=numcslocal+1;
                } else {
                    int connected_border=-1;
                    for ( jfc=0; jfc<  d_numconts[ind_b_fuera]; jfc++) {
                        int ind_c_fuera=jfc+num_max_conts*(ind_b_fuera);
                        int pf=d_ind_conts[ind_c_fuera].ini;
                        i_next_in=d_vec_conts[pf].act.i;
                        j_next_in=d_vec_conts[pf].act.j;
                        i_next_out=d_vec_conts[pf].ant.i;
                        j_next_out=d_vec_conts[pf].ant.j;
                        if ((i_con_dentro==i_next_out)&&(j_con_dentro==j_next_out)&&(i_con_fuera==i_next_in)&&(j_con_fuera==j_next_in)) {
                            connected_border=jfc; //jfc contour connects
                            pf_bueno=pf;
                            break;
                        }
                    }
                    if (connected_border==-1)

                        printf ("contorno no conectado:  bloquei %d, bloque j %d contorno %d, punto %d %d \n",ib*2, jb, jfc, i_con_dentro, j_con_dentro);
                    else {
                        contorno=connected_border;
                        d_vec_conts[p2].next=pf_bueno;
                        if ((i_con_dentro==i_antes)&&(j_con_dentro==j_antes)&&(i_con_fuera==i_con_ini)&&(j_con_fuera==j_con_ini)) {
                            //conected with begin of contour, so that closes
                            fin=1;

                            int contg=atomicAdd(d_numconts_glob,1); //store in global structure
                            d_ind_conts_glob[contg+1]=d_ind_conts[i_ind_conts];
                            i_ind_conts--;
                            numcslocal--;
                            d_ind_conts_glob[contg+1].fin=p2;
                            d_ind_conts_glob[contg+1].ini=nump1;
                            d_ind_conts_glob[contg+1].sts=CLOSED_CONTOUR;
                            marked[contorno+num_max_conts*(ind_b_fuera)]=-1;
                            marked[indcactual]=-1;

                        } else if (marked[contorno+num_max_conts*(ind_b_fuera)]>0) { //conecta con uno ya recorrido, pero no cierra
                            fin=1;

                            int numcontorno=marked[contorno+num_max_conts*(ind_b_fuera)]-1; //este es el numero de contorno de salida
                            d_ind_conts_out[numcontorno+indice_ini].ini=nump1;
                            marked[indcactual]=numcontorno+1;

                            numcslocal--;
                            i_ind_conts--;
                        } else {
                            int cont_siguiente= contorno+num_max_conts*(ind_b_fuera);
                            marked[cont_siguiente]=numcslocal+1;


                            i_salida=ibloque;
                            ibloque=ind_b_fuera;
                        }
                    }
                }
            }
        }
        //  }
    }
    ibloque=ibloque_abajo;
    for (int cont=0; cont< d_numconts[ibloque_abajo]; cont++) {
        ibloque=ibloque_abajo;
        int indcactual=cont+indice_ini_abajo;

        if(marked[indcactual]==0) { // if not marked as tracked, track it
            numcslocal=numcslocal+1;
            i_ind_conts++;

            d_ind_conts_out[i_ind_conts]=d_ind_conts[indcactual];
            nump1= d_ind_conts[indcactual].ini;
            p2=d_ind_conts[indcactual].fin;
            int i_con_dentro = d_vec_conts[p2].act.i;
            int j_con_dentro = d_vec_conts[p2].act.j;
            // next pixel, out of present block
            int i_con_fuera = d_vec_conts[p2].sig.i;
            int j_con_fuera = d_vec_conts[p2].sig.j;

            int ibl_fuera=i_con_fuera/numfbl; //coords of block of next pixel de bloque del punto siguiente
            if (ibl_fuera>mb-1)
                ibl_fuera=mb-1;
            int jbl_fuera=j_con_fuera/numcbl;
            if (jbl_fuera>nb-1)
                jbl_fuera=nb-1;
            int ind_b_fuera=ibl_fuera+mb*(jbl_fuera);
            if (ind_b_fuera!=ibloque_arriba) { //leaves by another block, added to list of open contours
                d_ind_conts_out[i_ind_conts].ini=nump1;
                d_ind_conts_out[i_ind_conts].fin=p2;
                d_ind_conts_out[i_ind_conts].sts=OPEN_CONTOUR;
            } else {
                connected_border=-1;
                for (jfc=0; jfc< d_numconts[ind_b_fuera]; jfc++) {
                    ind_c_fuera=jfc+num_max_conts*(ind_b_fuera);
                    int pf=d_ind_conts[ind_c_fuera].ini;
                    i_next_in=d_vec_conts[pf].act.i;
                    j_next_in=d_vec_conts[pf].act.j;
                    i_next_out=d_vec_conts[pf].ant.i;
                    j_next_out=d_vec_conts[pf].ant.j;
                    if ((i_con_dentro==i_next_out)&&(j_con_dentro==j_next_out)&&(i_con_fuera==i_next_in)&&(j_con_fuera==j_next_in)) {
                        connected_border=jfc; //jfc contour connects
                        pf_bueno=pf;
                        break;
                    }
                }

                if (connected_border==-1)

                    printf ("contorno no conectado:  bloquei %d, bloque j %d contorno %d, punto %d %d \n",ib*2, jb, jfc, i_con_dentro, j_con_dentro);
                else {
                    contorno=connected_border;
                    d_vec_conts[p2].next=pf_bueno;

                    int numcontorno=marked[contorno+num_max_conts*(ind_b_fuera)]-1;
                    d_ind_conts_out[numcontorno+indice_ini].ini=nump1;

                    marked[indcactual]=numcontorno+1;


                    numcslocal--;
                    i_ind_conts--;



                }
            }
        }
        //}
    }
    d_numconts_out[ibloque_sal_arriba]=numcslocal+1;
}

/**
* Kernel CUDA: horizontal_connection of borders . The closed borders are added to the global sructure
*
*/
__global__ void horizontal_connection(
    int num_max_conts,
    int     *d_numconts,
    VecCont *d_vec_conts,
    IndCont *d_ind_conts,
    int     *d_numconts_out,
    IndCont *d_ind_conts_out,
    int *marked,
    int numfbl,
    int numcbl,
    IndCont* d_ind_conts_glob,
    int* d_numconts_glob
) {
    //When this kernel is called , there will be only a row of blocks

    //int ib = blockIdx.x;
    int jb = blockIdx.y;
    int mb = gridDim.x;
    int nb = 2*gridDim.y;
    int i_con_ini, j_con_ini,i_antes, j_antes, i_salida, nump1,p2, jfc;
    int i_next_in, j_next_in, i_next_out,j_next_out, pf_bueno;
    int connected_border, ind_c_fuera, contorno,ind_b_fuera;
    int ibloque;
    int ibloque_izquierda=2*jb;
    int ibloque_sal_izquierda=jb;
    int ibloque_derecha=2*jb+1;
    int indice_ini=num_max_conts*(ibloque_izquierda);
    int indice_ini_derecha=num_max_conts*(ibloque_derecha);
    int numcslocal=-1;
    int i_ind_conts=numcslocal+indice_ini;
    for (int cont=0; cont< d_numconts[ibloque_izquierda]; cont++) {
        ibloque=ibloque_izquierda;
        int indcactual=cont+indice_ini;

        if (marked[indcactual]==0	) {
            numcslocal++;
            i_ind_conts++;

            d_ind_conts_out[i_ind_conts]=d_ind_conts[indcactual];
            int nump1=	d_ind_conts[indcactual].ini;
            i_con_ini=d_vec_conts[nump1].act.i;
            j_con_ini=d_vec_conts[nump1].act.j;
            i_antes=d_vec_conts[nump1].ant.i;
            j_antes=d_vec_conts[nump1].ant.j;

            int fin=0;
            contorno=cont;
            i_salida=ibloque_derecha;
            while(fin==0) {
                int indcaux=contorno+num_max_conts*(ibloque);
                // final point of contour
                p2 = d_ind_conts[indcaux].fin;

                // Coords of final point of contour, in present rectangle
                int i_con_dentro = d_vec_conts[p2].act.i;
                int j_con_dentro = d_vec_conts[p2].act.j;
                // next point of contour, out of present rectangle
                int i_con_fuera = d_vec_conts[p2].sig.i;
                int j_con_fuera = d_vec_conts[p2].sig.j;
                // Index of rectangle where next pixel is; in horizontal, ibl_siguiente is always zero

                int jbl_siguiente = j_con_fuera / numcbl ;
                if ( jbl_siguiente > nb-1 )
                    jbl_siguiente = nb-1;
                ind_b_fuera=mb*(jbl_siguiente);//indice de bloque siguiente
                int jfc;
                if (ind_b_fuera!=i_salida) {
                    fin =1;
                    d_ind_conts_out[i_ind_conts].ini=nump1;
                    d_ind_conts_out[i_ind_conts].fin=p2;
                    d_ind_conts_out[i_ind_conts].sts=OPEN_CONTOUR;
                    marked[indcactual]=numcslocal+1;
                } else {
                    int connected_border=-1;

                    for (jfc=0; jfc<  d_numconts[ind_b_fuera]; jfc++) {
                        ind_c_fuera=jfc+num_max_conts*(ind_b_fuera);
                        int pf=d_ind_conts[ind_c_fuera].ini;
                        i_next_in=d_vec_conts[pf].act.i;
                        j_next_in=d_vec_conts[pf].act.j;
                        i_next_out=d_vec_conts[pf].ant.i;
                        j_next_out=d_vec_conts[pf].ant.j;

                        if ((i_con_dentro==i_next_out)&&(j_con_dentro==j_next_out)&&(i_con_fuera==i_next_in)&&(j_con_fuera==j_next_in)) {
                            connected_border=jfc; //jfc contour connects
                            pf_bueno=pf;
                            break;
                        }
                    }

                    if (connected_border==-1)

                    {
                        printf ("contorno no conectado:  bloquei %d, bloque j %d contorno %d, punto %d %d \n",1, jb, jfc, i_con_dentro, j_con_dentro);
                    } else {

                        contorno=connected_border;
                        d_vec_conts[p2].next=pf_bueno;
                        if ((i_con_dentro==i_antes)&&(j_con_dentro==j_antes)&&(i_con_fuera==i_con_ini)&&(j_con_fuera==j_con_ini)) {
                            //connects with start of contour, so that closes
                            fin=1; //connected with end
                            int contg=atomicAdd(d_numconts_glob,1);
                            d_ind_conts_glob[contg+1]=d_ind_conts[i_ind_conts];
                            i_ind_conts--;
                            numcslocal--;
                            d_ind_conts_glob[contg+1].fin=p2;
                            d_ind_conts_glob[contg+1].ini=nump1;
                            d_ind_conts_glob[contg+1].sts=CLOSED_CONTOUR;
                            marked[contorno+num_max_conts*(ind_b_fuera)]=-1;
                            marked[indcactual]=-1;

                        } else if (marked[contorno+num_max_conts*(ind_b_fuera)]>0) { //connects with contour already tracked, but does not close
                            fin=1;
                            int numcontorno=marked[contorno+num_max_conts*(ind_b_fuera)]-1;
                            d_ind_conts_out[numcontorno+indice_ini].ini=nump1;
                            marked[indcactual]=numcontorno+1;


//El contorno i_ind_conts se ha integrado en el numcontorno, se disminuyen numcslocal y i_ind_conts

                            numcslocal--;
                            i_ind_conts--;
                        } else {
                            int cont_siguiente= contorno+num_max_conts*(ind_b_fuera);
                            marked[cont_siguiente]=numcslocal+1;

                            i_salida=ibloque;
                            ibloque=ind_b_fuera;
                        }
                    }
                }
            }
        }
        //   }
    }
    ibloque=ibloque_derecha;
    for (int cont=0; cont< d_numconts[ibloque_derecha]; cont++) {
        ibloque=ibloque_derecha;
        int indcactual=cont+indice_ini_derecha;

        if(marked[indcactual]==0) { // if not marked as tracked, track it
            numcslocal=numcslocal+1;
            i_ind_conts++;
            d_ind_conts_out[i_ind_conts]=d_ind_conts[indcactual];
            nump1= d_ind_conts[indcactual].ini;
            p2=d_ind_conts[indcactual].fin;
            int i_con_dentro = d_vec_conts[p2].act.i;
            int j_con_dentro = d_vec_conts[p2].act.j;
            // next point, out of rectangle
            int i_con_fuera = d_vec_conts[p2].sig.i;
            int j_con_fuera = d_vec_conts[p2].sig.j;


            int jbl_fuera=j_con_fuera/numcbl;
            if (jbl_fuera>nb-1)
                jbl_fuera=nb-1;
            int ind_b_fuera=mb*(jbl_fuera);
            if (ind_b_fuera!=ibloque_izquierda) {
                d_ind_conts_out[i_ind_conts].ini=nump1;
                d_ind_conts_out[i_ind_conts].fin=p2;
                d_ind_conts_out[i_ind_conts].sts=OPEN_CONTOUR;
            } else {
                connected_border=-1;
                for (jfc=0; jfc< d_numconts[ind_b_fuera]; jfc++) {
                    ind_c_fuera=jfc+num_max_conts*(ind_b_fuera);
                    int pf=d_ind_conts[ind_c_fuera].ini;
                    i_next_in=d_vec_conts[pf].act.i;
                    j_next_in=d_vec_conts[pf].act.j;
                    i_next_out=d_vec_conts[pf].ant.i;
                    j_next_out=d_vec_conts[pf].ant.j;
                    if ((i_con_dentro==i_next_out)&&(j_con_dentro==j_next_out)&&(i_con_fuera==i_next_in)&&(j_con_fuera==j_next_in)) {
                        connected_border=jfc; //jfc contour connects
                        pf_bueno=pf;
                        break;
                    }
                }

                if (connected_border==-1)

                    printf ("contorno no conectado:  bloquei %d, bloque j %d contorno %d, punto %d %d \n",2, jb, jfc, 0, j_con_dentro);
                else {
                    contorno=connected_border;
                    d_vec_conts[p2].next=pf_bueno;

                    int numcontorno=marked[contorno+num_max_conts*(ind_b_fuera)]-1;

                    d_ind_conts_out[numcontorno+num_max_conts*(ind_b_fuera)].ini=nump1;
                    marked[indcactual]=numcontorno+1;


                    numcslocal--;
                    i_ind_conts--;



                }
            }
        }
        //  }
    }
    d_numconts_out[ibloque_sal_izquierda]=numcslocal+1;
}


/**
 * plot contours obtained in a new image, GPU
 */
__global__ void plot_contours_gpu(
    IndCont *d_ind_conts,
    VecCont *d_vec_conts,
    uint8_t *d_Asal) {
    int i = blockIdx.x;

    // Position and coordinates of initial point
    int i_ind_conts_ini = d_ind_conts[i].ini;
    coord p = d_vec_conts[i_ind_conts_ini].act;

    // Position of final point
    int i_ind_conts_fin = d_ind_conts[i].fin;

    // write contour number in starting point
    d_Asal(p.i, p.j) = i + 1;

    // follow next points
    while ( i_ind_conts_ini != i_ind_conts_fin ) {
        i_ind_conts_ini = d_vec_conts[i_ind_conts_ini].next;
        p = d_vec_conts[i_ind_conts_ini].act;
        d_Asal(p.i, p.j) = i + 1;
    }

}


/**
 * plot contours obtained in a new image, CPU
 */
void plot_contours(
    int numcsg,
    IndCont *h_ind_conts,
    VecCont *h_vec_conts,
    uint8_t *h_Asal) {
    for( int i = 0; i < numcsg; i++ ) {
        // Position and coordinates of initial point
        int i_ind_conts_ini = h_ind_conts[i].ini;
        coord p = h_vec_conts[i_ind_conts_ini].act;

        // Position of final point
        int i_ind_conts_fin = h_ind_conts[i].fin;

        // write contour number in starting point
        h_Asal(p.i, p.j) = i + 1;

        // follow next points
        while ( i_ind_conts_ini != i_ind_conts_fin ) {
            i_ind_conts_ini = h_vec_conts[i_ind_conts_ini].next;
            p = h_vec_conts[i_ind_conts_ini].act;
            h_Asal(p.i, p.j) = i + 1;
        }
    }
}

//copy image to extended format
void copy_p_a_g(uint8_t *h_A, uint8_t *h_Ag) {


    for( int j = 0; j < N; j++ ) {
        for( int i = 0; i < M; i++ ) {
            h_Ag(i, j) = h_A(i,j);

        }
    }
}

//copy image form extended format to original format
void copy_g_a_p(uint8_t *h_Ag, uint8_t *h_A) {


    for( int j = 0; j < N; j++ ) {
        for( int i = 0; i < M; i++ ) {
            h_A(i, j) = h_Ag(i,j);

        }
    }
}

/**
 * Read image from .bin file
 */
void read4bin (uint8_t *h_A, const char fname[]) {
    uint8_t *h_Asal = (uint8_t*) malloc( M * N * sizeof(uint8_t) );

    FILE *fd;
    fd = fopen(fname, "r" );
    fread( (char*) h_Asal, sizeof(uint8_t), M * N, fd );
    fclose( fd );
    int k=0;
    for( int j = 0; j < N; j++ ) {
        for( int i = 0; i < M; i++ ) {
            h_A(i, j) = h_Asal[k];
            k++;
        }
    }
}


/**
 * output matrix to .bin file
 */
void write2bin (uint8_t *h_A, const char fname[]) {
    uint8_t *h_Asal = (uint8_t*) malloc( M * N * sizeof(uint8_t) );
    int k=0;
    for( int j = 0; j < N; j++ ) {
        for( int i = 0; i < M; i++ ) {
            h_Asal[k++] = h_A(i, j);
        }
    }

    FILE *fd;
    fd = fopen(fname, "w" );
    fwrite( (char*) h_Asal, sizeof(uint8_t), M * N, fd );
    fclose(fd);
}


/**
 * Output from small matrix to screen */
void printMatrix(uint8_t *h_A, const char info[]) {
    if ( (M < 20) && (N < 20) ) {
        printf("%s:\n   ", info);
        for( int j = 0; j < N; j++ )
            printf("%2d  ", j);
        printf("\n");
        for( int i = 0; i < M; i++ ) {
            printf("%d   ", i);
            for( int j = 0; j < N; j++ ) {
                printf("%d   ", h_A(i,j));
            }
            printf("\n");
        }
    }
}


/**
 * Output from matrix .bin fila and to screen
 */
void outputMatrix (uint8_t *h_A, const char name[]) {
    printMatrix(h_A, name);
    write2bin(h_A, name);
}

//kernel to ensure that outer limit of image is full of zeros
__global__ void borde_ceros(
    uint8_t *d_A
) {
    int i = threadIdx.x;
    while(i<M) {
        d_A[i]=0;
        d_A(i,(N - 1)) = 0;
        i=i+1024;
    }
    i = threadIdx.x;
    while(i<N) {

        d_A(0,i) = 0;
        d_A(M-1,i) = 0;
        i=i+1024;
    }

}



/**
 * Main program
*/
int main( int argc, char *argv[] ) {

    // number of repetitions, optional parameter, 20 is default
    int runs = 20;
    if ( argc > 1 ) {
        sscanf( argv[1], "%d", &runs);
        if( runs > 1000 ) {
            printf("Usage: %s [1..1000]\n",argv[0]);
            exit(-1);
        } else if ( runs > 1 ) {
            // if there are more than one , omit first
            runs++;
        }
    }

    // image
    printf("Imagen: %s\n", FILENAME);

    unsigned int mfb = N_BLOCKS_ROWS;
    unsigned int nfb = N_BLOCKS_COLS;

    int numblq = mfb * nfb;
    int numfbl = (Mg / mfb); //using Mg and Ng, exact divisions
    int numcbl = (Ng / nfb);


    unsigned int mat_mem_size  = sizeof(uint8_t) * M * N;
    unsigned int mat_mem_sizeg  = sizeof(uint8_t) * Mg * Ng;

    unsigned int blq_mem_size  = sizeof(int) * numblq;
    unsigned int vec_mem_size = sizeof(VecCont) * Mg * Ng*2;
    unsigned int ind_mem_size  = sizeof(IndCont)  * numblq * MAX_N_BORDS;
    unsigned int marked_mem_size  = sizeof(int)  * numblq * MAX_N_BORDS;
    unsigned int ind_mem_size_glob  = sizeof(IndCont)  * 100000;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Declaration of host variables
    startCPUTimer();
    uint8_t      *h_A            = (uint8_t*)      malloc( mat_mem_size );

    uint8_t      *h_Asal         = (uint8_t*)      malloc( mat_mem_size );
    uint8_t      *h_Ag            = (uint8_t*)      malloc( mat_mem_sizeg );
    uint8_t      *h_Asalg         = (uint8_t*)      malloc( mat_mem_sizeg );

    int      *h_numconts     = (int*)      malloc( blq_mem_size );
    int      *h_numconts_glob     = (int*)      malloc( sizeof(int) );
    VecCont *h_vec_conts   = (VecCont*) malloc( vec_mem_size );
    IndCont  *h_ind_conts    = (IndCont*)  malloc( ind_mem_size );
    IndCont  *h_ind_conts_glob    = (IndCont*)  malloc( ind_mem_size_glob );

    stopCPUTimer("malloc\t\t\t\t");


    // declaration of device variables
    int  *d_marked, *d_numconts_glob;
    uint8_t *d_A,*d_is_bord, *d_Asal;
    int      *d_numconts, *d_numconts_aux,*d_numconts_out;
    IndCont  *d_ind_conts, *d_ind_conts_aux, *d_ind_conts_out, *d_ind_conts_glob;
    VecCont *d_vec_conts;


    // memory for device variables
    startCudaTimer();
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_A,          mat_mem_sizeg  ), "cudaMalloc d_A" );
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_Asal,          mat_mem_size  ), "cudaMalloc d_Asal" );
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_marked,          marked_mem_size  ), "cudaMalloc d_marked" );
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_is_bord,    mat_mem_sizeg  ), "cudaMalloc d_es_count" );
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_numconts_glob,  sizeof(int)  ), "cudaMalloc d_numconts glob" );
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_numconts,   blq_mem_size  ), "cudaMalloc d_numconts" );
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_numconts_aux,   blq_mem_size  ), "cudaMalloc d_numconts" );
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_vec_conts, vec_mem_size ), "cudaMalloc d_vec_conts" );
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_ind_conts,  ind_mem_size  ), "cudaMalloc d_ind_conts" );
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_ind_conts_glob,  ind_mem_size_glob  ), "cudaMalloc d_ind_conts_glob" );
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_ind_conts_aux,  ind_mem_size  ), "cudaMalloc d_ind_conts" );

    stopCudaTimer("cudaMalloc\t\t\t");


    // reading image h_A
    startCPUTimer();
    read4bin ( h_A, FILENAME);
    memset(h_Ag, 0, mat_mem_sizeg);

//obtain extended matrix h_Ag
    copy_p_a_g(h_A,h_Ag);

    stopCPUTimer("readbin\t\t\t\t");
#ifdef OPENCV
    Mat img = imread(FILENAMEBMP);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
#endif
    // repeated executions
    for( int run = 0; run < runs; run++) {
//printf("run %d\n",run);

        // Memory device
        startCudaTimer();
        CUDA_SAFE_CALL( cudaMemcpy( d_A,      h_Ag,      mat_mem_sizeg, cudaMemcpyHostToDevice ), "cudaMemcpy d_A h_Ag" );
        CUDA_SAFE_CALL( cudaMemset( d_Asal,   0,  mat_mem_size  ), "cudaMemset d_Asal" );
        CUDA_SAFE_CALL( cudaMemset( d_is_bord,   0,  mat_mem_sizeg  ), "cudaMemset d_is_bord" );
        CUDA_SAFE_CALL( cudaMemset( d_marked,   0,  marked_mem_size  ), "cudaMemset d_marked" );

        CUDA_SAFE_CALL( cudaMemset( d_numconts,  0,  blq_mem_size  ), "cudaMemset d_numconts" );
        CUDA_SAFE_CALL( cudaMemset( d_numconts_aux,  0,  blq_mem_size  ), "cudaMemset d_numconts" );
        CUDA_SAFE_CALL( cudaMemset( d_numconts_glob,  0,  sizeof(int)  ), "cudaMemset d_numconts_glob" );
        CUDA_SAFE_CALL( cudaMemset( d_vec_conts, 0, vec_mem_size ), "cudaMemset d_vec_conts" );

        CUDA_SAFE_CALL( cudaMemset( d_ind_conts, 0, ind_mem_size ), "cudaMemset d_ind_conts" );
        CUDA_SAFE_CALL( cudaMemset( d_ind_conts_aux, 0, ind_mem_size ), "cudaMemset d_ind_conts" );
        CUDA_SAFE_CALL( cudaMemset( d_ind_conts_glob, 0, ind_mem_size_glob ), "cudaMemset d_ind_conts_glob" );

        saveCudaTimer(0, run);
        borde_ceros <<<1, 1024 >> > (d_A);

        memset(h_Asal, 0, mat_mem_size);

        dim3 dimBlock;
        dimBlock.x = THREADS_PER_BLOCK_X;
        dimBlock.y = THREADS_PER_BLOCK_Y;
        dim3 dimGrid;
        dimGrid.x = (M/dimBlock.x)+1;
        dimGrid.y = (N/dimBlock.y )+1;





        // Detection of contour points
        startCudaTimer();
        preprocessing_gpu<<< dimGrid, dimBlock >>>( d_A, d_is_bord );
        saveCudaTimer(2, run);


        // parallel tracking

        dimGrid.x = N_BLOCKS_ROWS;
        dimGrid.y = N_BLOCKS_COLS;

        dimBlock = 1;
        startCudaTimer();
        parallel_tracking<<< dimGrid, dimBlock >>>( d_A, d_is_bord, d_numconts, d_vec_conts, d_ind_conts, d_ind_conts_glob, d_numconts_glob );
        saveCudaTimer(3, run);

        numfbl = (Mg / mfb); //con Mg y Ng son divisiones exactas
        numcbl = (Ng / nfb);

        // Connectioon of contours of different rectangles
        startCudaTimer();
        int mbn=dimGrid.x/2;
        int num_max_c_etapa=MAX_N_BORDS;
//vertical connections
        while(mbn>=1) {
            dimGrid.x=mbn;

            vertical_connection<<< dimGrid, dimBlock >>>(num_max_c_etapa,d_numconts,d_vec_conts,d_ind_conts,d_numconts_aux,d_ind_conts_aux,d_marked,numfbl,numcbl, d_ind_conts_glob, d_numconts_glob );
            cudaDeviceSynchronize();
            CUDA_SAFE_CALL( cudaMemset( d_marked, 0, marked_mem_size ), "cudaMemset d_ind_conts" );
            d_numconts_out=d_numconts;
            d_numconts=d_numconts_aux;
            d_numconts_aux=d_numconts_out;
            CUDA_SAFE_CALL( cudaMemset( d_numconts_aux,  0,  blq_mem_size  ), "cudaMemset d_numconts" );
            d_ind_conts_out=d_ind_conts;
            d_ind_conts=d_ind_conts_aux;
            d_ind_conts_aux=d_ind_conts_out;
            CUDA_SAFE_CALL( cudaMemset( d_ind_conts_aux, 0, ind_mem_size ), "cudaMemset d_ind_conts" );
            num_max_c_etapa=num_max_c_etapa*2;
            numfbl=numfbl*2;
            mbn=mbn/2;
        }
        dimGrid.x=1;
        int nbn=dimGrid.y/2;
//horizontal connections
        while(nbn>=1) {
            dimGrid.y=nbn;
            horizontal_connection<<< dimGrid, dimBlock >>>(num_max_c_etapa,d_numconts,d_vec_conts,d_ind_conts,d_numconts_aux,d_ind_conts_aux,d_marked,numfbl,numcbl, d_ind_conts_glob, d_numconts_glob);
            cudaDeviceSynchronize();
            CUDA_SAFE_CALL( cudaMemset( d_marked, 0, marked_mem_size ), "cudaMemset d_ind_conts" );
            d_numconts_out=d_numconts;
            d_numconts=d_numconts_aux;
            d_numconts_aux=d_numconts_out;
            CUDA_SAFE_CALL( cudaMemset( d_numconts_aux,  0,  blq_mem_size  ), "cudaMemset d_numconts" );
            d_ind_conts_out=d_ind_conts;
            d_ind_conts=d_ind_conts_aux;
            d_ind_conts_aux=d_ind_conts_out;
            CUDA_SAFE_CALL( cudaMemset( d_ind_conts_aux, 0, ind_mem_size ), "cudaMemset d_ind_conts" );
            num_max_c_etapa=num_max_c_etapa*2;
            nbn=nbn/2;
            numcbl=numcbl*2;
        }

        saveCudaTimer(4, run);
        CUDA_SAFE_CALL( cudaMemcpy( h_numconts_glob,  d_numconts_glob,  sizeof(int), cudaMemcpyDeviceToHost ), "cudaMemcpy numconts d_numconts glob" );

        plot_contours_gpu<<<h_numconts_glob[0]+1,1>>> (d_ind_conts_glob, d_vec_conts, d_Asal );
        startCudaTimer();
        CUDA_SAFE_CALL( cudaMemcpy( h_Asal, d_Asal, mat_mem_size, cudaMemcpyDeviceToHost ), "cudaMemcpy h_Asal d_Asal" );
        saveCudaTimer(5, run);
        // Copy contours from device to host; thi

        //CUDA_SAFE_CALL( cudaMemcpy( h_numconts,  d_numconts,  blq_mem_size, cudaMemcpyDeviceToHost ), "cudaMemcpy numconts d_numconts" );

        /*CUDA_SAFE_CALL( cudaMemcpy( h_vec_conts, d_vec_conts, vec_mem_size, cudaMemcpyDeviceToHost ), "cudaMemcpy h_vec_conts d_vec_conts" );
        //CUDA_SAFE_CALL( cudaMemcpy( h_ind_conts, d_ind_conts, ind_mem_size, cudaMemcpyDeviceToHost ), "cudaMemcpy h_ind_conts d_ind_conts" );
        CUDA_SAFE_CALL( cudaMemcpy( h_ind_conts_glob, d_ind_conts_glob, ind_mem_size_glob, cudaMemcpyDeviceToHost ), "cudaMemcpy h_ind_conts_glob d_ind_conts_glob" );
            saveCudaTimer(5, run);

        // plot contours
        // startCPUTimer();

        plot_contours( h_numconts_glob[0]+1, h_ind_conts_glob, h_vec_conts, h_Asal );
        // saveCPUTimer(6, run); */
#ifdef OPENCV
        saveCPUTimer(0, run);
        startCPUTimer();
        vector<vector<Point> > contours0;
        findContours( gray, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        saveCPUTimer(1, run);
#endif
        /*contours.resize(contours0.size());

          for( size_t k = 0; k < contours0.size(); k++ )
              approxPolyDP(Mat(contours0[k]), contours[k], 3, true);
          namedWindow( "contours", 1 );
           Mat cnt_img = Mat::zeros(1028, 1232, CV_8UC3);
          int _levels = 3;
          drawContours( cnt_img, contours, _levels <= 0 ? 3 : -1, Scalar(128,255,255),
                        3, LINE_AA, hierarchy, std::abs(_levels) );

          imshow("contours", cnt_img);*/
    }

    averageCudaTimer( 0, runs, "cudaMemcpy and cudaMemset\t");
    averageCudaTimer( 2, runs, "kernel preprocessing_gpu\t");
    averageCudaTimer( 3, runs, "kernel parallel_tracking\t");
    averageCudaTimer( 4, runs, "kernels connection");
    averageCudaTimer( 5, runs, "cudaMemcpyDeviceToHost\t\t");
    //averageCPUTimer(  0, runs, "conexion_contornos\t\t");
    //averageCPUTimer(  6, runs, "plot_contours\t\t");
#ifdef OPENCV
    averageCPUTimer(  1, runs, "opencv\t\t");
#endif
    // Salida de la matriz inicial y del resultado final
    startCPUTimer();
    printMatrix( h_A, "h_A" );
    outputMatrix( h_Asal, "h_Asal.bin" );


    stopCPUTimer("output\t\t\t\t");
    printf(" number of borders GPU %d \n",(int)h_numconts_glob[0]);
#ifdef OPENCV
    vector<vector<Point> > contours0;
    findContours( gray, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    printf("\n number borders opencv %d \n",(int)contours0.size());
#endif

    // Liberar memoria de dispositivo
    startCudaTimer();
    CUDA_SAFE_CALL( cudaFree( d_A          ), "cudaFree d_A" );
    CUDA_SAFE_CALL( cudaFree( d_Asal          ), "cudaFree d_Adal" );
    CUDA_SAFE_CALL( cudaFree( d_is_bord    ), "cudaFree d_is_bord" );

    CUDA_SAFE_CALL( cudaFree( d_numconts   ), "cudaFree d_numconts" );
    CUDA_SAFE_CALL( cudaFree( d_vec_conts ), "cudaFree d_vec_contsA" );

    CUDA_SAFE_CALL( cudaFree( d_ind_conts  ), "cudaFree d_ind_conts" );

    stopCudaTimer("cudaFree\t\t\t");

    // Liberar memoria de host
    startCPUTimer();
    free( h_A );
    free( h_Asal );
    free( h_numconts );
    free( h_vec_conts );
    free( h_ind_conts );

    stopCPUTimer("free\t\t\t\t");

    exit(EXIT_SUCCESS);
}
