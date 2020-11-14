#ifndef RESAMPLERDP_H
#define RESAMPLERDP_H

#include <vector>
#include "lines.h"
#include "mergeLines.h"

void resampleStrokesRDP(double epsilon, std::vector<Line>& lines);
std::vector<Eigen::VectorXd> RDP(double epsilon, std::vector<Eigen::VectorXd>::iterator it_begin, std::vector<Eigen::VectorXd>::iterator it_end);
#endif