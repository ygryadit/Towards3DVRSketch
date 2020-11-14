#include "resampleRDP.h"
#include <vector>
#include "lines.h"
#include "mergeLines.h"

std::vector<Eigen::VectorXd> RDP(double epsilon, std::vector<Eigen::VectorXd>::iterator it_begin, std::vector<Eigen::VectorXd>::iterator it_end)
{
	double d_max = 0;
	double d = 0;
	std::vector<Eigen::VectorXd>::iterator it_add;
	std::vector<Eigen::VectorXd> points_left, points_right, points_final;
	
	double projected_point_t;

	// Find the point with the maximum distance
	if ((it_end - it_begin) > 1)
	{
		for (std::vector<Eigen::VectorXd>::iterator it = it_begin + 1; it != it_end; ++it)
		{
			d =  distancePtSegment((*it_begin), (*it_end), (*it), projected_point_t);

			if (d > d_max) {
				it_add = it;
				d_max = d;
			}
		}
	}

	// If max distance is greater than epsilon, recursively simplify
	if (d_max > epsilon) {
		// Recursive call
		points_left = RDP(epsilon, it_begin, it_add);
		points_right = RDP(epsilon, it_add, it_end);

		// Build the result list
		points_final.assign(points_left.begin(), points_left.end() - 1);
		points_final.insert(points_final.end(), points_right.begin(), points_right.end());
	}
	else {
		points_final.push_back(*it_begin);
		points_final.push_back(*it_end);
	}
	// Return the result
	return points_final;

}

//An implementation of Ramer-Douglas-Peucker
void resampleStrokesRDP(double epsilon, std::vector<Line>& lines)
{
	Eigen::VectorXd v_last;
	for (int i = 0; i < static_cast<int>(lines.size()); ++i)
	{
	
		if (*(lines[i].begin()) != *(lines[i].end() - 1))
			lines[i] = RDP(epsilon, lines[i].begin(), lines[i].end()-1);
		else 
		{
			v_last = *(lines[i].end() - 1);
			lines[i] = RDP(epsilon, lines[i].begin(), lines[i].end() - 2);
			lines[i].push_back(v_last);
		}
	}
}
