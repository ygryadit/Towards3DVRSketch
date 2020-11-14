#ifndef WRITEOBJCHAINS_H
#define WRITEOBJCHAINS_H
#include "lines.h"
#include <string>
#include <Eigen/Core>
#include <vector>

//At the very end, there can be a list of chains.Each line starting with
//\# chain" is a chain. The following numbers are the indexes of the
//edges forming the chain.
bool writeOBJChains(
    const std::string& str,
    const Eigen::MatrixXd& V,
    const std::vector<std::vector<int> >& Edges,
    const std::vector< std::vector<int>>& chains);

void converLines2Chains(const std::vector<Line>& lines,
    Eigen::MatrixXd& V,
    std::vector<std::vector<int>>& Edges,
    std::vector< std::vector<int>>& chains);


#endif