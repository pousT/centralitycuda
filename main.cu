#include <stdio.h>
#include <string.h>
#include "cuBCStruct.h"
#include "constant.h"
#include "graph_indexed.h"
#include "timing.cu"
#include "BC_gpu.cu"

#define CPU  2
#define GPU  4
#define ERROR 16
#define DIRECTED  5
#define UNDIRECTED  6

// parse the input arguments
void parse(int argc, char * argv[],
           int & mode,
           int & graph,
           char * filename[] );

int main(int argc, char * argv[])
{
   int mode = GPU;
   int graph = UNDIRECTED;
   char * filename[3] = {NULL, NULL, NULL};
   parse(argc, argv, mode, graph, filename);

   GraphIndexed* pGraph = new GraphIndexed();
   if(graph == UNDIRECTED) {
     if(!pGraph->LoadUndirect(filename[0]))
     {
        return -1;
     }
   } else {
     if(!pGraph->LoadDirect(filename[0]))
     {
        return -1;
     }
   }
   printf("num edges: %d\n", pGraph->NumberOfEdges());


   printf("Initial graph and bc data on CPU\n");
   cuGraph* pCUGraph = NULL;
   cuBC*    pBCData  = NULL;
   initGraph(pGraph, pCUGraph);
   initBC(pCUGraph, pBCData);
   cuGraph* pGPUCUGraph = NULL;
   cuBC*    pGPUBCData  = NULL;
   if(mode&GPU) {
      printf("Initial graph and bc data on GPU\n");
      initGPUGraph(pCUGraph, pGPUCUGraph);
      initGPUBC(pBCData, pGPUBCData);
   }


   std::string bcfile(filename[0]);
   bcfile = bcfile.substr(0, bcfile.length()-5);

   printf("Start computing BC\n");
   // Start timing
   StopWatchInterface *total_timer = NULL;
    startTimer(&total_timer);

   switch (mode) {
     case GPU:
     {
       gpuComputeBCOpt(pGPUCUGraph, pGPUBCData);
       bcfile.append(".gpu_bc");
     }
     case CPU:
     {
       //do cpu calculation
       cpuComputeBCOpt(pCUGraph, pBCData);
       bcfile.append(".cpu_bc");
     }
   }

    printf("Total time: %f (ms)\n", endTimer(&total_timer));
    if(mode&GPU) {
    copyBackGPUBC(pGPUBCData, pBCData);
    }
    cpuSaveBC(pGraph, pBCData, bcfile.c_str());

    if(mode&GPU) {
      freeGPUGraph(pGPUCUGraph);
      freeGPUBC(pGPUBCData);
    }

   freeGraph(pCUGraph);
   freeBC(pBCData);
   delete pGraph;

   return 0;
}


void parse(int argc, char * argv[],
            int & mode,
            int & graph,
            char * filename[])
{
    for(int i=0; i<argc; i++)
    {
        if(strcmp(argv[i], "-gpu")==0 ||
           strcmp(argv[i], "-g")==0)
        {
            mode = GPU;
        }
        else if(strcmp(argv[i], "-cpu")==0 ||
           strcmp(argv[i], "-c")==0)
        {
            mode = CPU;
        }
        else if(strcmp(argv[i], "-error")==0 ||
           strcmp(argv[i], "-e")==0)
        {
            if(i+2<argc)
            {
               mode = ERROR;
               i++;
               filename[1] = argv[i];
               i++;
               filename[2] = argv[i];
            }
        }
        else if(strcmp(argv[i], "-file")==0 ||
           strcmp(argv[i], "-f")==0)
        {
            i++;
            if(i<argc)
                filename[0] = argv[i];
        }
        else if(strcmp(argv[i], "-d")==0 ||
           strcmp(argv[i], "-directed")==0)
         {
           graph = DIRECTED;
         }
         else if(strcmp(argv[i], "-u")==0 ||
            strcmp(argv[i], "-undirected")==0)
          {
            graph = UNDIRECTED;
          }
        else if(strcmp(argv[i], "-help")==0 ||
           strcmp(argv[i], "--help")==0)
        {
            printf("cudaRaytracer [options] -f graph_file\n"
                    "options:\n"
                    "   -gpu,  -g  : running program on GPU\n"
                    "   -cpu,  -c  : running program on CPU\n"
                    "   -d, -directed: read in file as directed graph\n"
                    "   -u, -undirected: read in file as undirected graph\n"
                );
            exit(0);
        }
    }

   if(!filename[0])
   {
       printf("cudaRaytracer [options] -f graph_file\n"
              "options:\n"
              "   -gpu,  -g  : running program on GPU\n"
              "   -cpu,  -c  : running program on CPU\n"
              "   -d, -directed: read in file as directed graph\n"
              "   -u, -undirected: read in file as undirected graph\n"

            );
       exit(0);
   }
}
