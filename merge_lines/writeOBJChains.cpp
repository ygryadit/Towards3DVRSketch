#include "writeOBJChains.h"
#include <igl/list_to_matrix.h>

#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <cassert>


bool writeOBJChains(
    const std::string& str,
    const Eigen::MatrixXd& V,
    const std::vector<std::vector<int> >& Edges,
    const std::vector< std::vector<int>>& chains)
{
    using namespace std;
    using namespace Eigen;
    
    assert(V.cols() == 3 && "V should have 3 columns");
    
    ofstream s(str);
    if (!s.is_open())
    {
        fprintf(stderr, "IOError: writeOBJ() could not open %s\n", str.c_str());
        return false;
    }
    s << V.format(IOFormat(FullPrecision, DontAlignCols, " ", "\n", "v ", "", "", "\n"));

    for (const auto& edge : Edges)
    {
        int edge_size = edge.size();
        assert(edge_size != 0);

        s << (edge_size == 2 ? "l" : "f");

        for (const auto& vi : edge)
        {
            s << " " << vi;
        }
        s << "\n";
    }

    for (const auto& chain : chains)
    {
        int chain_size = chain.size();

        if (chain_size == 0)
            continue;

        s << "# chain";

        for (const auto& ei : chain)
        {
            s << " " << ei;
        }
        s << "\n";
    }

    return true;
}



void converLines2Chains(const std::vector<Line>& lines,
    Eigen::MatrixXd& V,
    std::vector<std::vector<int>>& Edges,
    std::vector< std::vector<int>>& chains)
{
    std::vector<Eigen::VectorXd> V_;    
    std::vector<int> edge;
    std::vector<int> chain;
    int num_vertices; 

    for (auto line : lines)
    {
        chain.clear();
        num_vertices = line.size();
        V_.push_back(line[0]);
        for (int i = 1; i < (num_vertices-1); i++)
        {
            edge.push_back(V_.size());
            V_.push_back(line[i]);
            edge.push_back(V_.size());
            Edges.push_back(edge);
            chain.push_back(Edges.size());
            edge.clear();
        }
        if (line[0] != line[num_vertices-1])
        {
            edge.push_back(V_.size());
            V_.push_back(line[num_vertices-1]); //add the last vertex
            edge.push_back(V_.size());
            Edges.push_back(edge);
            chain.push_back(Edges.size());
            edge.clear();
        }
        else
        {
            edge.push_back(V_.size());
            edge.push_back(V_.size()-num_vertices+2); //closed curve put the first vertex of the chain
            Edges.push_back(edge);
            chain.push_back(Edges.size());
            edge.clear();
        }
        chains.push_back(chain);
    }

    //
    ////igl::list_to_matrix(V_, V);

    int num_vertices_all = V_.size();
    V.resize(num_vertices_all, 3);
    for (int i = 0; i < num_vertices_all; i++)
    {
        V.row(i) = V_[i];
    }
}