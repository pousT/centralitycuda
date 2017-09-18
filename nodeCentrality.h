#ifndef _NODE_CENTRALITY_H_
#define _NODE_CENTRALITY_H_
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
class nodeCentrality
{
public:
    int id;
    int dc; //degree centrality
    float bc; // btw centrality
    float inverse_cc; // inverse cc
public:
    nodeCentrality(int index) {
        id = index;
        dc = 0;
        bc = 0.0;
        inverse_cc = 0.0;
    }
 };
#endif //_NODE_H_
