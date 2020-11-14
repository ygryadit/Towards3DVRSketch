#ifndef readEdges_h
#define readEdges_h

#include <sstream>
#include <Eigen/Core>

#include "lines.h"

bool readEdges(const std::string obj_file_name,
				Eigen::MatrixXd& V1_,
				Eigen::MatrixXd& V2_,
				std::vector<Line>& lines);


#endif