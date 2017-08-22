#ifndef _GRAPH_INDEXED_H_
#define _GRAPH_INDEXED_H_

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

class GraphIndexed
{
public:
  typedef std::vector<int> Nodes;
  typedef std::vector<int> NodeEdge;
  typedef std::map<int, int> NodeIndex;
  typedef std::vector<NodeEdge> NodeEdges;
  typedef std::vector<std::pair<int, int> > Edges;

protected:
  Nodes m_Nodes; // node ids
  Edges m_Edges; // edge vector <from id, end id>
  NodeEdges m_NodeEdges;
  NodeIndex m_NodeIndexes; // node positions in m_Nodes <id, pos in m_Nodes>
  bool  m_bGraphModified;

public:
  GraphIndexed()
  {
    m_bGraphModified = false;
  }

public:
  virtual bool LoadUndirect(const std::string& filename)
  {
    if (filename.rfind(".edge") == (filename.length() - 5)) // read edeges
      {
      std::ifstream is;
          is.open(filename.c_str());

          //check
          if(!is.good())
          {
              std::cout << " Graph failed to load: " << filename << std::endl;
              return false;
          }

          int nnodes, nedges, id1, id2;
      // number of nodes and edges in the graph
          is >> nnodes >> nedges;
          while((is >> id1 >> id2) && !is.eof())
          {
              AddEdgeUndirect(id1, id2); // undireced edge
          }

          is.close();
    }

    std::cout<< "Graph loaded successfully: " << filename << std::endl;
    return true;
  }
  bool LoadDirect(const std::string& filename)
  {
    if (filename.rfind(".edge") == (filename.length() - 5)) // read edeges
      {
      std::ifstream is;
          is.open(filename.c_str());

          //check
          if(!is.good())
          {
              std::cout << " Graph failed to load: " << filename << std::endl;
              return false;
          }

          int nnodes, nedges, id1, id2;
      // number of nodes and edges in the graph
          is >> nnodes >> nedges;
          while((is >> id1 >> id2) && !is.eof())
          {
              AddEdgeDirect(id1, id2);
          }

          is.close();
    }
public:
  int AddEdgeUndirect(int nd1, int nd2)
  {
    int idx1,idx2;
    //current id of edge
    int eg = (int)m_Edges.size(); // edge id

    NodeIndex::iterator iter;
    // if node1 not the the graph
    if((iter=m_NodeIndexes.find(nd1))==m_NodeIndexes.end())
    {
      idx1 = (int)m_Nodes.size();
      // add node into node list
      m_Nodes.push_back(nd1);
      // store node position
      m_NodeIndexes[nd1] = idx1;
      //edge list
      std::vector<int> tmp;
      tmp.push_back(eg);
      m_NodeEdges.push_back(tmp); // vector of edge ids
    }
    else
    {
      // node1 already in the list
      idx1 = (*iter).second;
      m_NodeEdges[idx1].push_back(eg); // edges connected to m_Nodes[idx1] to its neighbours
    }
    eg += 1;
    // if node2 not in the graph
    if((iter=m_NodeIndexes.find(nd2))==m_NodeIndexes.end())
    {
      idx2 = (int)m_Nodes.size();
      m_Nodes.push_back(nd2);
      m_NodeIndexes[nd2] = idx2;
      std::vector<int> tmp;
      tmp.push_back(eg);
      m_NodeEdges.push_back(tmp);
    }
    else
    {
      idx2 = (*iter).second;
      m_NodeEdges[idx2].push_back(eg); // add edges to node 2
    }
    // new edge1
    std::pair<int, int> tmp1(idx1, idx2);
    m_Edges.push_back(tmp1);
    // new edge2
    std::pair<int, int> tmp2(idx2, idx1);
    m_Edges.push_back(tmp2);


    return eg;
  }

  int AddEdgeDirect(int nd1, int nd2)
  {
    int idx1,idx2;
    //current id of edge
    int eg = (int)m_Edges.size(); // edge id

    NodeIndex::iterator iter;
    // if node1 not the the graph
    if((iter=m_NodeIndexes.find(nd1))==m_NodeIndexes.end())
    {
      idx1 = (int)m_Nodes.size();
      // add node into node list
      m_Nodes.push_back(nd1);
      // store node position
      m_NodeIndexes[nd1] = idx1;
      //edge list
      std::vector<int> tmp;
      tmp.push_back(eg);
      m_NodeEdges.push_back(tmp); // vector of edge ids
    }
    else
    {
      // node1 already in the list
      idx1 = (*iter).second;
      m_NodeEdges[idx1].push_back(eg); // edges connected to m_Nodes[idx1] to its neighbours
    }
    // if node2 not in the graph
    if((iter=m_NodeIndexes.find(nd2))==m_NodeIndexes.end())
    {
      idx2 = (int)m_Nodes.size();
      m_Nodes.push_back(nd2);
      m_NodeIndexes[nd2] = idx2;
      std::vector<int> tmp;
      m_NodeEdges.push_back(tmp);
    }
    else
    {
      idx2 = (*iter).second;
    }
    // new edge
    std::pair<int, int> tmp(idx1, idx2);
      m_Edges.push_back(tmp);

    return eg;
  }

  int NumberOfNodes(void)const
  { return (int)m_Nodes.size();}
    int NumberOfEdges(void)const
  { return (int)m_Edges.size();}

  const Nodes & GetNodes()const
  { return m_Nodes; }
  const Nodes & GetNodes()
  { return m_Nodes; }

  const Edges & GetEdges()const
  { return m_Edges; }
  const Edges & GetEdges()
  { return m_Edges; }

  const NodeEdge & GetEdges(int n)
  { return m_NodeEdges[n]; }
  const NodeEdge & GetEdges(int n)const
  { return m_NodeEdges[n]; }

  virtual Nodes  GetNodes(int n)
  {
    const NodeEdge & nes = GetEdges(n);
    NodeEdge::const_iterator iter;
    Nodes nds;
    for(iter = nes.begin(); iter!=nes.end(); iter++ )
    {
      int nd2 = m_Edges[(*iter)].second;
      nds.push_back(nd2);
    }
    return nds;
  }
  //given id of a node, return its neighbours
  Nodes  GetNodes(int n)const
  {
    const NodeEdge & nes = GetEdges(n);
    NodeEdge::const_iterator iter;
    Nodes nds;
    for(iter = nes.begin(); iter!=nes.end(); iter++ )
    {
      int nd2 = m_Edges[(*iter)].second;
      nds.push_back(nd2);
    }
    return nds;
  }
  // give id of two nodes
  virtual int GetEdge(int n1, int n2)
  {
    const NodeEdge & nes = GetEdges(n1);
    NodeEdge::const_iterator iter;
    for(iter = nes.begin(); iter!=nes.end(); iter++ )
    {
      int nd2 = m_Edges[(*iter)].second;
      if(nd2 == n2)
        return (*iter);
    }
    return -1;
  }
};

#endif //_GRAPH_INDEXED_H_
