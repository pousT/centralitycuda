#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <helper_timer.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "cuBCStruct.h"
#include "device_functions.h"
//#include "sm_12_atomic_functions.h"
//#include "sm_21_atomic_functions.h"

#define BLOCK_SIZE 768

__global__ void cuda_computeInverseCC_block(const cuGraph graph,
                                     const node_list srcs, float* inverse_CCs)
{
   __shared__ cuBC  bcData;
   __shared__ float toprocess;
   __shared__ int   edge2;
   __shared__ float inverseCC;
   if(threadIdx.x==0)
   {
      bcData = const_BCDatas[blockIdx.x];
      edge2  = (bcData.nedge<<1);
   }
   int   * pNumSPs     = const_BCDatas[blockIdx.x].numSPs;
   float * pDependency = const_BCDatas[blockIdx.x].dependency;
   __syncthreads();

   for(int src_idx = blockIdx.x; src_idx < srcs.size; src_idx += NUM_BLOCKS)
   //int src_idx = blockIdx.x;
   {
      // clear data
      for(int node_idx = threadIdx.x; node_idx < bcData.nnode; node_idx += blockDim.x)
      {
         pNumSPs[node_idx] = 0;
         pDependency[node_idx] = 0.f;
         bcData.distance[node_idx] = -1;
      }
      for(int node_idx = threadIdx.x; node_idx < edge2; node_idx += blockDim.x)
      {
         bcData.successor[node_idx] = false;
      }

      __syncthreads();

      // initial BFS
      if(threadIdx.x==0)
      {
         int node = srcs.nodes[src_idx];
         inverseCC = 0;
         // bcData.numSPs[node] = 1;
         bcData.distance[node] = 0;
         toprocess = 1;
         int nb_cur = graph.index_list[node];
         int nb_end = graph.index_list[node+1];
         inverseCC += nb_end - nb_cur;

         for(; nb_cur<nb_end; nb_cur++)
         {
            node = graph.edge_node2[nb_cur];
            bcData.distance[node] = 1;
         }
      }

      int distance = 0;
      __syncthreads();
      float inverse_distance = 1;
      // BFS
      while(toprocess>0)
      {
         __syncthreads();
         toprocess = 0;
         distance ++;
         __syncthreads();

         for(int edge_idx = threadIdx.x; edge_idx < edge2; edge_idx += blockDim.x)
         {
            int from = graph.edge_node1[edge_idx];
            if(bcData.distance[from] == distance)
            {
               int to          = graph.edge_node2[edge_idx];
               int to_distance = bcData.distance[to];
               if(to_distance<0)
               {
                  bcData.distance[to] = to_distance = distance+1;
                  toprocess =1;

               }
               inverse_distance = 1/to_distance;
               if(to_distance>distance)
               {
                  // add inverse of distance to inverseCC
                  AtomicAdd(&inverseCC, inverse_distance);
               }
            }
         }

         __syncthreads();
      }
      inverse_CCs[src_idx] = inverseCC;
   }
}

void gpuComputeInverseCCOpt(const cuGraph * pGraph, cuBC * pBCData)
{
   // init bc data for each block
   cuBC pCPUBCDatas[NUM_BLOCKS];
   cuBC * pTmp[NUM_BLOCKS];
   pCPUBCDatas[0] = *pBCData;
   int distances[pGraph->nnode][pGraph->nnode];
   checkCudaErrors(cudaMalloc ((void **) &(srcs.nodes), sizeof(int)*pGraph->nnode));
   //declare array store inverse closeness centrality
   float * inverse_CC[pGraph->nnode];
   for(int i=1; i<NUM_BLOCKS; i++)
   {
      pTmp[i] = NULL;
      initGPUBC(pBCData, pTmp[i]);
      pCPUBCDatas[i] = *pTmp[i];
   }
   cudaMemcpyToSymbol(const_BCDatas, pCPUBCDatas, sizeof(cuBC)*NUM_BLOCKS);

   // prepare source node list
   node_list srcs;
   srcs.size = pGraph->nnode;
   checkCudaErrors(cudaMalloc ((void **) &(srcs.nodes), sizeof(int)*pGraph->nnode));
   int * tmp = (int*)calloc(pGraph->nnode, sizeof(int));
   for(int i=0; i<pGraph->nnode; i++) tmp[i] = i;
   checkCudaErrors(cudaMemcpy(srcs.nodes, tmp, sizeof(int)*pGraph->nnode, cudaMemcpyHostToDevice));
   free(tmp);

   StopWatchInterface *kernel_timer = NULL;
    startTimer(&kernel_timer);

   // call kernels
   cuda_computeBC_block<<<NUM_BLOCKS, BLOCK_SIZE>>>(*pGraph, srcs, inverse_CCs);


   printf("Kernel time: %f (ms)\n", endTimer(&kernel_timer));

   checkCudaErrors(cudaFree(srcs.nodes));
   for(int i=1; i<NUM_BLOCKS; i++)
      freeGPUBC(pTmp[i]);
}