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
      GraphIndexed::Nodes neighbors = pGraph->GetNodes(i);
      GraphIndexed::NodeEdge edges  = pGraph->GetEdges(i);
      GraphIndexed::Nodes::iterator iter1 = neighbors.begin();
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
void initDirectGraph(const GraphDirected * pGraph, cuGraph *& pCUGraph)
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
      GraphIndexed::Nodes neighbors = pGraph->GetNodes(i);
      GraphIndexed::NodeEdge edges  = pGraph->GetEdges(i);
      GraphIndexed::Nodes::iterator iter1 = neighbors.begin();
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
      memset(pBCData->distance, 0xff, pBCData->nnode*sizeof(int));
      memset(pBCData->successor, 0, pBCData->nedge*sizeof(bool));
      // do not clear nodeBC & edgeBC which is accumulated through iterations
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

void cpuSaveBC(const GraphIndexed * pGraph, const cuBC * pBCData, const char* filename)
{
   std::ofstream of(filename);
   of << "Version" << std::endl;
   of << 2 << std::endl;
#ifdef COMPUTE_EDGE_BC
   of << pBCData->nnode << " " << pBCData->nedge << std::endl;
#else
   of << pBCData->nnode << " " << 0 << std::endl;
#endif
   of << "0  0" << std::endl;
   const GraphIndexed::Nodes & nodes = pGraph->GetNodes();
   for(int i=0; i<pBCData->nnode; i++)
   {
      of << nodes[i] << "\t" << pBCData->nodeBC[i] << std::endl;
   }
#ifdef COMPUTE_EDGE_BC
   for(int i=0; i<pBCData->nedge; i++)
   {
      of << i << "\t" << pBCData->edgeBC[i] << std::endl;
   }
#endif
   of.close();
}

void cpuSaveBC(const cuBC * pBCData, const char* filename)
{
   std::ofstream of(filename);
   for(int i=0; i<pBCData->nnode; i++)
   {
      of << i << "\t" << pBCData->nodeBC[i] << std::endl;
   }
#ifdef COMPUTE_EDGE_BC
   for(int i=0; i<pBCData->nedge; i++)
   {
      of << i << "\t" << pBCData->edgeBC[i] << std::endl;
   }
#endif
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
