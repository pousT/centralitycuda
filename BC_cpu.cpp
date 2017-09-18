#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <queue>
#include <vector>
#include "cuBCStruct.h"

void initGraph(const GraphIndexed * pGraph, cuGraph *& pCUGraph)
{
   if(pCUGraph)
      freeGraph(pCUGraph);

   pCUGraph = (cuGraph*)calloc(1, sizeof(cuGraph));
   pCUGraph->nnode = pGraph->NumberOfNodes();
   pCUGraph->nedge = pGraph->NumberOfEdges();

   pCUGraph->edge_node1 = (int*)calloc(pCUGraph->nedge, sizeof(int));
   pCUGraph->edge_node2 = (int*)calloc(pCUGraph->nedge, sizeof(int));
   pCUGraph->index_list = (int*)calloc(pCUGraph->nnode+1, sizeof(int));
#ifdef COMPUTE_EDGE_BC
   pCUGraph->edge_id = (int*)calloc(pCUGraph->nedge, sizeof(int));
   int* edge_id = pCUGraph->edge_id;
#endif

   int offset = 0;
   int* edge_node1 = pCUGraph->edge_node1;
   int* edge_node2 = pCUGraph->edge_node2;
   int* index_list = pCUGraph->index_list;

   for(int i=0; i<pGraph->NumberOfNodes(); i++)
   {
#ifdef COMPUTE_EDGE_BC
      GraphIndexed::NodeIds neighbors = pGraph->GetNodes(i);
      GraphIndexed::NodeEdge edges  = pGraph->GetEdges(i);
      GraphIndexed::NodeIds::iterator iter1 = neighbors.begin();
      GraphIndexed::NodeEdge::iterator iter2 = edges.begin();
      for(; iter1!=neighbors.end(); iter1++, iter2++)
      {
         *edge_node1++ = i;
         *edge_node2++ = (*iter1);
         *edge_id++ = (*iter2);
      }
#else
      GraphIndexed::Nodes neighbors = pGraph->GetNodes(i);
      GraphIndexed::Nodes::iterator iter1;
      for(iter1=neighbors.begin(); iter1!=neighbors.end(); iter1++)
      {
         *edge_node1++ = i;
         *edge_node2++ = (*iter1);
      }
#endif
      *index_list++ = offset;
      offset += neighbors.size();
   }
   *index_list = offset;
}

void freeGraph(cuGraph *& pGraph)
{
   if(pGraph)
   {
      free(pGraph->edge_node1);
      free(pGraph->edge_node2);
      free(pGraph->index_list);
#ifdef COMPUTE_EDGE_BC
      free(pGraph->edge_id);
#endif
      free(pGraph);
      pGraph = NULL;
   }
}

void initBC(const cuGraph * pGraph, cuBC *& pBCData)
{
   if(pBCData)
      freeBC(pBCData);

   pBCData = (cuBC*)calloc(1, sizeof(cuBC));
   pBCData->nnode = pGraph->nnode;
   pBCData->nedge = pGraph->nedge;

   pBCData->numSPs = (int*)calloc(pBCData->nnode, sizeof(int));
   pBCData->dependency = (float*)calloc(pBCData->nnode, sizeof(float));
   pBCData->inverseCC = (float*)calloc(pBCData->nnode, sizeof(float));
   pBCData->distance = (int*)calloc(pBCData->nnode, sizeof(int));
   pBCData->nodeBC = (float*)calloc(pBCData->nnode, sizeof(float));
   pBCData->successor  = (bool*)calloc(pBCData->nedge, sizeof(bool));
#ifdef COMPUTE_EDGE_BC
   pBCData->edgeBC = (float*)calloc(pBCData->nedge, sizeof(float));
#endif
}

void freeBC(cuBC *& pBCData)
{
   if(pBCData)
   {
      free(pBCData->successor);
      free(pBCData->numSPs);
      free(pBCData->nodeBC);
      free(pBCData->dependency);
      free(pBCData->inverseCC);
      free(pBCData->distance);
#ifdef COMPUTE_EDGE_BC
      free(pBCData->edgeBC);
#endif
      free(pBCData);
      pBCData = NULL;
   }
}

void clearBC(cuBC * pBCData)
{
   if(pBCData)
   {
      pBCData->toprocess = 0;
      memset(pBCData->numSPs, 0, pBCData->nnode*sizeof(int));
      memset(pBCData->dependency, 0, pBCData->nnode*sizeof(float));
      memset(pBCData->inverseCC, 0, pBCData->nnode*sizeof(float));
      memset(pBCData->distance, 0xff, pBCData->nnode*sizeof(int));
      memset(pBCData->successor, 0, pBCData->nedge*sizeof(bool));
      // do not clear nodeBC & edgeBC which is accumulated through iterations
   }
}

void cpuSaveBC(const GraphIndexed * pGraph, const cuBC * pBCData, const char* filename)
{
   std::ofstream of(filename);
#ifdef COMPUTE_EDGE_BC
   of << pBCData->nnode << " " << pBCData->nedge << std::endl;
#else
   of << pBCData->nnode << " " << 0 << std::endl;
#endif
   of << "0  0" << std::endl;
   const GraphIndexed::Nodes & nodes = pGraph->GetNodes();
   for(int i=0; i<pBCData->nnode; i++)
   {
      of << nodes[i].id << "\t" << pBCData->nodeBC[i] << std::endl;
   }
#ifdef COMPUTE_EDGE_BC
   for(int i=0; i<pBCData->nedge; i++)
   {
      of << i << "\t" << pBCData->edgeBC[i] << std::endl;
   }
#endif
   of.close();
}

void cpuSaveBC(GraphIndexed * pGraph, const cuBC * pBCData)
{
   const GraphIndexed::Nodes & nodes = pGraph->GetNodes();
   for(int i=0; i<pBCData->nnode; i++)
   {
      pGraph->setBC(i, pBCData->nodeBC[i]);
      pGraph->setInvCC(i, pBCData->inverseCC[i]);
   }
}

void cpuSaveBCfile(const GraphIndexed * pGraph, const char* filename)
{
   std::ofstream of(filename);
  const GraphIndexed::Nodes & nodes = pGraph->GetNodes();
   for(int i=0; i<nodes.size(); i++)
   {
      of << nodes[i].id << "\t" << nodes[i].dc <<"\t"<<nodes[i].bc <<"\t" <<nodes[i].inverse_cc << std::endl;
   }

   of.close();
}

void cpuLoadBC(const cuBC * pBCData, const char* filename)
{
   if(!pBCData)
      return;

   std::ifstream inf(filename);
   int id;
   for(int i=0; i<pBCData->nnode; i++)
   {
      inf >> id >> pBCData->nodeBC[i];
   }
   inf.close();
}


//cpu calculation
int  cpuBFSOpt(const cuGraph * pGraph, cuBC * pBCData, int startNode, std::vector<int> & traversal)
{
   pBCData->numSPs[startNode] = 1;
   pBCData->distance[startNode] = 0;
   pBCData->toprocess = 1;
   int distance  = 0;
   int index = 0;
   std::deque<int> fifo;
   fifo.push_back(startNode);
   while(!fifo.empty())
   {
      int from = fifo.front();
      fifo.pop_front();
      traversal[index++] = from;

      distance = pBCData->distance[from];

      int nb_cur = pGraph->index_list[from];
      int nb_end = pGraph->index_list[from+1];
      for(; nb_cur<nb_end; nb_cur++)
      {
         int nb_id = pGraph->edge_node2[nb_cur];
         int nb_distance = pBCData->distance[nb_id];

         if(nb_distance<0)
         {
            pBCData->distance[nb_id] = nb_distance = distance+1;
            fifo.push_back(nb_id);
         }

         if(nb_distance>distance)
         {
            pBCData->successor[nb_cur] = true;
             pBCData->numSPs[nb_id] += pBCData->numSPs[from];
         }
      }
   }
   return distance;
}

int  cpuBFSOpt(const cuGraph * pGraph, cuBC * pBCData, int startNode, std::vector<int> & traversal, int wavefrontLmt)
{
   pBCData->numSPs[startNode] = 1;
   pBCData->distance[startNode] = 0;
   pBCData->toprocess = 1;
   int distance  = 0;
   int lastdist  = 0;
   int index = 0;
   std::deque<int> fifo;
   fifo.push_back(startNode);
   while(!fifo.empty())
   {
      int from = fifo.front();
      fifo.pop_front();
      traversal[index++] = from;

      lastdist = distance;
      distance = pBCData->distance[from];
      if(distance!=lastdist && fifo.size()>wavefrontLmt)
      {
         traversal.resize(index-1);
         return distance;
      }

      int nb_cur = pGraph->index_list[from];
      int nb_end = pGraph->index_list[from+1];
      for(; nb_cur<nb_end; nb_cur++)
      {
         int nb_id = pGraph->edge_node2[nb_cur];
         int nb_distance = pBCData->distance[nb_id];

         if(nb_distance<0)
         {
            pBCData->distance[nb_id] = nb_distance = distance+1;
            fifo.push_back(nb_id);
         }
         if(nb_distance>distance)
         {
            pBCData->numSPs[nb_id] += pBCData->numSPs[from];
            pBCData->successor[nb_cur] = true;
         }
      }
   }
   return distance;
}

void cpuUpdateBCOpt(const cuGraph * pGraph, cuBC * pBCData, int distance, const std::vector<int> & traversal)
{
   std::vector<int>::const_reverse_iterator criter;
   for(criter=traversal.rbegin(); criter!=traversal.rend(); criter++)
   {
      int from = (*criter);

      if(pBCData->distance[from]>=distance)
         continue;

      int nb_cur = pGraph->index_list[from];
      int nb_end = pGraph->index_list[from+1];
      float numSPs = pBCData->numSPs[from];
      float dependency = 0;
      for(; nb_cur<nb_end; nb_cur++)
      {
         if(pBCData->successor[nb_cur])
         {
            int nb_id = pGraph->edge_node2[nb_cur];

            float partialDependency = numSPs / pBCData->numSPs[nb_id];
            partialDependency *= (1.0f + pBCData->dependency[nb_id]);

            dependency += partialDependency;
            int edgeid = pGraph->edge_id[nb_cur];
            pBCData->edgeBC[edgeid] += partialDependency;
         }
      }
      pBCData->dependency[from] = dependency;
      pBCData->nodeBC[from] += dependency;
   }
}

void cpuHalfBC(cuBC * pBCData)
{
   for(int i=0; i<pBCData->nnode; i++)
      pBCData->nodeBC[i] *= 0.5f;

#ifdef COMPUTE_EDGE_BC
   for(int i=0; i<pBCData->nedge; i++)
      pBCData->edgeBC[i] *= 0.5f;
#endif
}

// cpu optimized version
void cpuComputeBCOpt(const cuGraph * pGraph, cuBC * pBCData)
{
   for(int i=0; i<pGraph->nnode; i++)
   {
      clearBC(pBCData);
      std::vector<int> traversal;
      traversal.resize(pGraph->nnode);
      int distance = cpuBFSOpt(pGraph, pBCData, i, traversal);
      float bk = pBCData->nodeBC[i];
      cpuUpdateBCOpt(pGraph, pBCData, distance, traversal);
      pBCData->nodeBC[i] = bk;
   }

   cpuHalfBC(pBCData);
}
