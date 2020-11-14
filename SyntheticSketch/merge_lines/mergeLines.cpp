
#include "mergeLines.h"
#include "resampleRDP.h"
#include "writeOBJChains.h"
#include <sstream>
#include <iostream>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort
#include <igl/list_to_matrix.h>
#include <igl/writeOBJ.h>


using namespace std;

extern string obj_filename;
double dist_threshold;

//#define DEBUG true


template <typename T>
vector<size_t> sort_indexes(const vector<T>& v) {

	// initialize original index locations
	vector<size_t> idx(v.size());
	iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	// using std::stable_sort instead of std::sort
	// to avoid unnecessary index re-orderings
	// when v contains elements of equal values 
	stable_sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

	return idx;
}

double returnLinesMergeThreshold(double l1, double l2, double thr)
{
	return min(max(thr * l1, thr * l2), dist_threshold);
}

void removeShortLines(std::vector<Line>& lines, std::vector<double>& lines_lengths, const double length_threshold) {
	double length_l;

	
	lines_lengths.clear();

	for (int i = 0; i < lines.size(); i++)
	{
		length_l = computeLineLength(lines[i]);
		if (length_l < length_threshold)
		{
			lines.erase(lines.begin() + i);
			i--;
		}
		else
		{
			lines_lengths.push_back(length_l);
		}
	}
}

double computeLineLength(const Line& line) 
{
	double length_l = 0;

	for (int i = 1; i < line.size(); i++)
	{
		length_l += (line[i] - line[i - 1]).norm();
	}

	return length_l;
}

void saveLines(const vector<Line> &lines, string name)
{
	std::vector<std::vector<int>> Edges;
	Eigen::MatrixXd V;
	std::vector<std::vector<int>> chains;

	/*linesToVerticesEdges(lines,
		Edges,
		V);*/

	string filename = obj_filename;
	filename.replace(filename.end() - 4, filename.end(), "_" + name + ".obj");
	cout << filename << endl;
	

	converLines2Chains(lines,
		V,
		Edges,
		chains);


	writeOBJChains(filename, V, Edges, chains);

	//igl::writeOBJ(filename, V, Edges);

}


void mergeLines(std::vector<Line>& lines, 
				Eigen::MatrixXd& V1_,
				Eigen::MatrixXd& V2_,
				double threshold)
{
	double max_dim = findMaxDim(V1_, V2_);

	dist_threshold = threshold * max_dim;
	double dist_threshold_line;
	// In a greedy way find a distance to all other lines
	double dist_average, dist_average_;

	int num_files = 0;
	std::vector<double> lines_lengths;

	cout << "before " << lines.size() << endl;
	removeShortLines(lines, lines_lengths, dist_threshold);
	cout << "after" << lines.size() << endl;

	#ifdef DEBUG
		saveLines(lines, "short_lines_removed");
	#endif  

	resampleStrokesRDP(dist_threshold/5.0, lines);

	#ifdef DEBUG
		saveLines(lines, "resampled");
	#endif  

	threshold = 0.05;

	for (int i = 0; i < lines.size(); i++)
	{
		//std::cout << "i = " << i << "/ " << lines.size() << endl;
		std::vector<int> lines_inds, lines_inds_;
		std::vector<double> distances, distances_;

		if (num_files == 6)
			cout << num_files << endl;

		//Find all lines closer than a threshold:
		findDistancesToAllOtherLines(lines, lines_lengths, i, threshold, lines_inds_, distances_);
		
		//Sort according to a distance:
		for (auto i : sort_indexes(distances_))
		{
			distances.push_back(distances_[i]);
			lines_inds.push_back(lines_inds_[i]);
		}
		
		//
		bool restart = false;
		for (auto j : lines_inds)			
		{
			if (restart)
				break;
			//std::cout << "j = " << j << "/ " << lines_inds.size() << endl;
			std::vector<double> seg_l1, seg_l1_;
			std::vector<double> seg_l2, seg_l2_;
			double dist_end_points;

	
			findCommonRegion(lines[i], lines[j], seg_l1, seg_l2, dist_end_points);
			seg_l1_ = seg_l1;
			seg_l2_ = seg_l2;
	

			dist_threshold_line = returnLinesMergeThreshold(lines_lengths[i], lines_lengths[j], threshold); 

			if (dist_end_points > dist_threshold_line)
				continue;

			Line l_aggreagte, l_aggreagte_;
			Line l1_ = lines[i], l2_ = lines[j];

			mergeCommonRegion(lines[i], lines[j], seg_l1, seg_l2,
				l_aggreagte,
				dist_average);

			
			//mergeCommonRegion(l1_, l2_, seg_l1_, seg_l2_,
			//		l_aggreagte_,
			//		dist_average_);			
			//
			//if (dist_average_ > dist_average)
			//{
			//	l_aggreagte = l_aggreagte_;
			//	seg_l1 = seg_l1_;
			//	seg_l2 = seg_l2_;
			//	lines[i] = l1_;
			//	lines[j] = l2_;
			//	dist_average = dist_average_;
			//}


			//cout << "dist_average = " << dist_average << endl;


			//mergeCommonRegion(lines[j], lines[i], seg_l2, seg_l1,
			//	l_aggreagte,
			//	dist_average);

			//cout << "dist_average = " << dist_average << endl;

			

			if (dist_average < dist_threshold_line)
			{
				std::cout << "Merged: " << i << ' ' << j << std::endl;

				//l_aggreagte = RDP(dist_threshold_line / 4.0, l_aggreagte.begin(), l_aggreagte.end() - 1);


				//Save lines to merge:
				vector<Line> linesMerged;
				linesMerged.push_back(lines[i]);
				linesMerged.push_back(lines[j]);

				#ifdef DEBUG
					saveLines(linesMerged, to_string(num_files) + "_to_merge_i" + to_string(i) + "_j" + to_string(j));
				#endif
				if (l_aggreagte.size() > 1)
				{
					if (*(l_aggreagte.begin()) != *(l_aggreagte.end() - 1))
						l_aggreagte = RDP(dist_threshold_line / 4.0, l_aggreagte.begin(), l_aggreagte.end() - 1);
					else
					{
						Eigen::VectorXd v_last;
						v_last = *(l_aggreagte.end() - 1);
						l_aggreagte = RDP(dist_threshold_line / 4.0, l_aggreagte.begin(), l_aggreagte.end() - 2);
						l_aggreagte.push_back(v_last);
					}
				}


				//Save aggregate before concatenation
				vector<Line> linesAggreagte;
				linesAggreagte.push_back(l_aggreagte);
				
				#ifdef DEBUG
					saveLines(linesAggreagte, to_string(num_files) + "_aggregated_common");
				#endif

				aggregateCurve(lines[i], lines[j], seg_l1, seg_l2, l_aggreagte);

				linesAggreagte.clear();
				linesAggreagte.push_back(l_aggreagte);

				#ifdef DEBUG
					saveLines(linesAggreagte, to_string(num_files++) + "_aggregated");
				#endif

				lines[i] = l_aggreagte;
				lines_lengths[i] = computeLineLength(l_aggreagte);
				//lines[i] = lines[j];
				lines.erase(lines.begin() + j);
				lines_lengths.erase(lines_lengths.begin() + j);
				i--;
				
				#ifdef DEBUG
					saveLines(lines, to_string(num_files++) + "_all");
				#endif

				restart = true;
				break;
			}
			
			//if (distLineLine(lines[i], lines[j], dist) & (dist < threshold))
			//{
			//	// Check if the distance is less than a certain thresold, if so merge or remove one of the lines.
			//	
			//}
		}
	}	
}

void findDistancesToAllOtherLines(std::vector<Line>& lines, std::vector<double>& lines_lengths, 
	int ind_l_cur, const double dist_threshold, std::vector<int>& lines_inds, std::vector<double>& distances)
{
	//Find all the lines that are tangenitlaly aliggned in the closest point and revert the order of the lines point to be similarily directed at the closest point.
	double dist_;
	bool aligned_;
	double dist_threshold_line;

	for (int i = ind_l_cur+1; i < (lines.size() - 1); i++)
	{
		if (i == ind_l_cur)
		{
			continue;
		}

		if ((ind_l_cur == 2) & (i == 76) )
			cout << ind_l_cur << endl;


		dist_threshold_line = returnLinesMergeThreshold(lines_lengths[ind_l_cur], lines_lengths[i], dist_threshold);

		distancePolylinePolyline(lines[ind_l_cur], lines[i], dist_threshold_line, dist_, aligned_);

		if (aligned_)
		{			
			//cout << "ind_l_cur " << ind_l_cur << "i " << i << endl;
			distances.push_back(dist_);
			lines_inds.push_back(i);
		}
	}
}

void distancePolylinePolyline(const Line l1, Line& l2, const double dist_threshold,  double& distance, bool& aligned)
{
	double seg_temp;
	double dist_;
	distance = std::numeric_limits<double>::max();

	aligned = false;

	for (int i = 0; i < (l1.size() - 1); i++)
	{	
		dist_ = distPointPolyline(l1[i], l2, seg_temp);
		if (dist_ < distance)
		{
			
			if (dist_ < dist_threshold)
			{
				aligned = checkAlignement(l1, l2, i, seg_temp);
				//aligned = true;

				if (aligned)
					distance = dist_;
			}
		}
	}
}


bool checkAlignement(const Line l1, Line& l2, const int pi, const double seg_temp)
{
	Eigen::VectorXd dir1, dir2;

	//First direction
	if (pi == 0)
	{
		//first segment
		dir1 = l1[pi + 1] - l1[pi];
		//dir1 = l1[pi + 2] - l1[pi];
	}
	else if (pi == (l1.size() - 1))
	{
		//last segment
		dir1 = l1[pi] - l1[pi-1];		
		//dir1 = l1[pi] - l1[pi - 2];		
	}
	else
	{
		dir1 = 0.5*( (l1[pi] - l1[pi - 1]) + (l1[pi + 1] - l1[pi]) );
	}
	
	dir1 = dir1 / dir1.norm();


	//Second direction
	int s1, s2;
	s1 = floor(seg_temp);
	s2 = ceil(seg_temp);
	double rem = seg_temp - s1;

	if (s1 == 0)
	{
		//first segment
		dir2 = l2[s1 + 1] - l2[s1];
		//dir2 = l2[s1 + 2] - l2[s1];
	}
	else if (s2 == (l2.size() - 1))
	{
		//last segment
		dir2 = l2[s2] - l2[s2 - 1];
		//dir2 = l2[s2] - l2[s2 - 2];
	}
	else if (rem < 1e-5)
	{
		//coincides with one of the vertices:
		dir2 = 0.5 * ((l2[s1] - l2[s1 - 1]) + (l2[s1 + 1] - l2[s1]));
	}
	else 
	{
		//in the middle of the segment
		dir2 = l2[s2] - l2[s1];
	}

	dir2 = dir2 / dir2.norm();


	//Dot product
	double cos_dist = dir1.dot(dir2);

	if (abs(cos_dist) > 0.99)
	{
		//Invert the line:
		if (cos_dist < 0)
		{			
			//cout << "inverted " << endl;
			invertLine(l2);
		}

		return true;
	}
	else
		return false;//return false;
}


void invertLine(Line& l)
{
	Eigen::VectorXd temp;
	int num_points = l.size()-1;
	/*for (int i = 0; i < l.size(); i++)
	{
		cout << l[i] << endl;
	}
	cout << "---" << endl;*/
	for (int i = 0; i < floor(l.size()/2.0); i++)
	{
		temp = l[i];		
		l[i] = l[num_points - i];
		l[num_points - i] = temp;		
	}

	//for (int i = 0; i < l.size(); i++)
	//{
	//	cout << l[i] << endl;
	//}
}

bool isClosedCurve(Line l)
{
	Eigen::VectorXd v1 = l[0];
	Eigen::VectorXd v2 = l[l.size()-1];
	return (v1 - v2).norm() < 1e-5;
}

void findCommonRegionClosedOpen(const Line& lclosed, const Line& lopen, std::vector<double>& seg_l1, std::vector<double>& seg_l2)
{
	seg_l1.push_back(0);
	seg_l1.push_back(0);

	distPointPolyline(lopen[0], lclosed, seg_l1[0]);
	distPointPolyline(lopen[lopen.size() - 1], lclosed, seg_l1[1]);
	seg_l2.push_back(0);
	seg_l2.push_back(lopen.size() - 1);
}


void findCommonRegion(Line l1, Line l2, std::vector<double>& seg_l1, std::vector<double>& seg_l2, double& dist)
{
	if (isClosedCurve(l1) & isClosedCurve(l2))
	{
		seg_l1.push_back(0);
		seg_l1.push_back(l1.size()-1);
		seg_l2.push_back(0);
		seg_l2.push_back(l2.size() - 1);
		return;
	}
	else if (isClosedCurve(l1) & !isClosedCurve(l2))
	{
		findCommonRegionClosedOpen(l1, l2, seg_l1, seg_l2);
		return;
	}
	else if ((!isClosedCurve(l1)) & isClosedCurve(l2)) 
	{
		findCommonRegionClosedOpen(l2, l1, seg_l2, seg_l1);
		return;
	}

	Eigen::VectorXd distances(4,1);

	double seg2_1, seg2_2, seg1_1, seg1_2;
	dist = 0;
	distances[0] = distPointPolyline(l1[0], l2, seg2_1);
	distances[1] = distPointPolyline(l1[l1.size() - 1], l2, seg2_2);
	distances[2] = distPointPolyline(l2[0], l1, seg1_1);
	distances[3] = distPointPolyline(l2[l2.size() - 1], l1, seg1_2);

	//std::cout << "=============" << std::endl;

	//std::cout << l1[0] << std::endl;
	//std::cout << l1[l1.size() - 1] << std::endl;
	//std::cout << l2[0] << std::endl;
	//std::cout << l2[l2.size() - 1] << std::endl;

	//std::cout << "______________" << std::endl;

	//std::cout << distances[0] << std::endl;
	//std::cout << distances[1] << std::endl;
	//std::cout << distances[2] << std::endl;
	//std::cout << distances[3] << std::endl;



	Eigen::VectorXd distances_sorted(4, 1);
	distances_sorted = distances;
	std::sort(distances_sorted.data(), distances_sorted.data() + distances_sorted.size());


	//std::cout << distances_sorted[0] << std::endl;
	//std::cout << distances_sorted[1] << std::endl;
	//std::cout << distances_sorted[2] << std::endl;
	//std::cout << distances_sorted[3] << std::endl;

	int i1, i2;
	if (distances[0] < distances[2])
		i1 = 0;
	else
		i1 = 2;

	if (distances[1] < distances[3])
		i2 = 1;
	else
		i2 = 3;

	dist = 0.5 * (distances[i1] + distances[i2]);

	if ((i1 == 0) | (i2 == 0))
	{
		seg_l1.push_back(0);
		seg_l2.push_back(seg2_1);
	}
	
	if ((i1 == 1) | (i2 == 1))
	{
		seg_l1.push_back(l1.size() - 1);
		seg_l2.push_back(seg2_2);
	}
	
	if ((i1 == 2) | (i2 == 2))
	{
		seg_l1.push_back(seg1_1);
		seg_l2.push_back(0);
	}
	
	if ((i1 == 3) | (i2 == 3))
	{
		seg_l1.push_back(seg1_2);
		seg_l2.push_back(l2.size() - 1);
	}
}

double distPointPolyline(Eigen::VectorXd p, Line line, double& seg_ind)
{
	double dist, dist_min = std::numeric_limits<double>::max();
	double projected_point_t;

	//cout << "p = " << p << endl;

	for (int i = 1; i < line.size(); i++)
	{
		dist = distancePtSegment(line[i-1], line[i], p, projected_point_t);
		if (dist < dist_min)
		{
			dist_min = dist;
			seg_ind = (i-1) + projected_point_t;
			//cout << "line[i-1] = " << line[i - 1] << endl;
			//cout << "line[i] = " << line[i] << endl;
			//Line line_temp(line.begin() + (i - 1), line.begin() + i+1);
		/*	cout << " line_temp[0] = " << line_temp[0] << endl;
			cout << " line_temp[1] = " << line_temp[1] << endl;*/
			//Eigen::VectorXd p_temp = seg_num_2_vertex(seg_ind, line_temp);
			/*std::cout << "------- " <<  std::endl;
			std::cout << "p_temp = " << p_temp << std::endl;
			std::cout << "------- " <<  std::endl;*/
		}
	}
	//std::cout << "dist_min = " << dist_min << std::endl;
	return dist_min;
}


double distancePtSegment(Eigen::VectorXd& a, Eigen::VectorXd b, Eigen::VectorXd p, double& projected_point_t)
{
	
	//cout << " b = " << b << endl;
	//cout << " a = " << a << endl;
	//cout << " p = " << p << endl;

	Eigen::VectorXd n = b - a;
	Eigen::VectorXd pa = p - a;

	projected_point_t = (pa.dot(n) / n.dot(n));
	Eigen::VectorXd c = a + n * projected_point_t;

	
	Eigen::VectorXd d;

	if (projected_point_t < 0)
	{
		projected_point_t = 0;
		c = a;		
	}
	else if (projected_point_t > 1.0)
	{
		projected_point_t = 1.0;
		c = b;	
	}
	

	d = p - c;
	//std::cout << "------- " << std::endl;
	//cout << " c = " << c << endl;
	//std::cout << "------- " << std::endl;

	//std::cout << "projected_point_t = " << projected_point_t << std::endl;
	return d.norm();
}


bool reverseTheOrderIfNeeded(Line& l, std::vector<double> &seg_l)
{
	
	if (seg_l[0] > seg_l[1])
	{		
		invertLine(l);

		seg_l[1] = l.size() - 1 - seg_l[1];
		seg_l[0] = l.size() - 1 - seg_l[0];
	
		//cout << "" << endl;
		return true;
	}	
	return false;
}

void aggregateCurve(	Line &l1, 
						Line &l2,
						std::vector<double>& seg_l1,
						std::vector<double>& seg_l2,
						Line& l_aggreagte)
{

	//cout << "l1" << endl;
	//for (int i = 0; i < l1.size(); i++)
	//{
	//	cout << l1[i] << endl;
	//}

	//cout << "l2" << endl;
	//for (int i = 0; i < l2.size(); i++)
	//{
	//	cout << l2[i] << endl;
	//}
	//
	//cout << "l_aggreagte" << endl;
	//for (int i = 0; i < l_aggreagte.size(); i++)
	//{
	//	cout << l_aggreagte[i] << endl;
	//}

	double rel_seg_l1[2];
	double rel_seg_l2[2];

	if (isClosedCurve(l1) & isClosedCurve(l2))
	{
		return;
	}

	if (isClosedCurve(l1) & (!isClosedCurve(l2)))
	{
		l_aggreagte = l1;
		return;
	}

	if (isClosedCurve(l2) & (!isClosedCurve(l1)))
	{
		l_aggreagte = l2;
		return;
	}

	// Reverse the order of points in the lines to be aligned:
	bool reverse_line;
	reverse_line = reverseTheOrderIfNeeded(l1, seg_l1);
	reverseTheOrderIfNeeded(l2, seg_l2);

	if (reverse_line)
		invertLine(l_aggreagte);

	// Relative indices
	rel_seg_l1[0] = seg_l1[0] / (l1.size()-1);
	rel_seg_l1[1] = seg_l1[1] / (l1.size()-1);

	rel_seg_l2[0] = seg_l2[0] / (l2.size()-1);
	rel_seg_l2[1] = seg_l2[1] / (l2.size()-1);


	// Concatenate:
	double eps = 1e-10;
	Line l_aggreagte_;
	//begin
	if ((rel_seg_l1[0] > rel_seg_l2[0]) & (rel_seg_l1[0] > eps))
	{
		Line temp(l1.begin(), l1.begin() + ceil(seg_l1[0]));
		l_aggreagte_ = temp;
	}
	else if ((rel_seg_l1[0] < rel_seg_l2[0]) & (rel_seg_l2[0] > eps))
	{
		Line temp(l2.begin(), l2.begin() + ceil(seg_l2[0]));
		l_aggreagte_ = temp;
	}
	//common
	l_aggreagte_.insert(l_aggreagte_.end(), l_aggreagte.begin(), l_aggreagte.end());
	 
	//end
	if ((rel_seg_l1[1] < rel_seg_l2[1]) & (rel_seg_l1[1] <  (1- eps)))
	{
		Line temp(l1.begin() + ceil(seg_l1[1]), l1.end());
		l_aggreagte_.insert(l_aggreagte_.end(), temp.begin(), temp.end());
	}
	else if ((rel_seg_l1[1] > rel_seg_l2[1])& (rel_seg_l2[1] < (1 - eps)))
	{
		Line temp(l2.begin() + ceil(seg_l2[1]), l2.end());
		l_aggreagte_.insert(l_aggreagte_.end(), temp.begin(), temp.end());
	}

	l_aggreagte = l_aggreagte_;
}

Eigen::VectorXd seg_num_2_vertex(double seg_num, const Line& line)
{
	//cout << "begin = " << line[floor(seg_num)] << endl;
	//cout << "step  = " << line[ceil(seg_num)] - line[floor(seg_num)] << endl;
	return line[floor(seg_num)] + (seg_num - floor(seg_num)) * (line[ceil(seg_num)] - line[floor(seg_num)]);
}

void mergeCommonRegion(Line& l1,
	Line& l2,
	std::vector<double> seg_l1,
	std::vector<double> seg_l2,
	Line& l_aggreagte,
	double &dist_average)
{
	// Go over the points: find the closest - return the middle - compute average dist and alignement.

	//int num_points = std::max(ceil(seg_l2[1] - seg_l2[0]), ceil(seg_l1[1] - seg_l1[0]));
	int num_points;
	if (seg_l1[1] < seg_l1[0])
	{
		double temp = seg_l1[1];
		seg_l1[1] = seg_l1[0];
		seg_l1[0] = temp;
	/*	invertLine(l1);
		invertLine(l2);*/
		temp = seg_l2[1];
		seg_l2[1] = seg_l2[0];
		seg_l2[0] = temp;
		//cout << " inverted segments order " << endl;
	}

	num_points = ceil(seg_l1[1]) - floor(seg_l1[0])+1;
	
	
	//Fill in the segment numbers:
	std::vector<float> seg_nums;
	seg_nums.push_back(seg_l1[0]);

	for (int i = 1; i < (num_points-1); i++)
		seg_nums.push_back(floor(seg_l1[0])+i);
	
	seg_nums.push_back(seg_l1[1]);

	
	//Compute an average curve and distances:

	Eigen::VectorXd p1, p2;
	std::vector<double> distances;
	double seg_ind_1, seg_ind_2;

	for (int i = 0; i < num_points; i++)
	{
		p1 = seg_num_2_vertex(seg_nums[i], l1);
		//std::cout << "p1 = " << p1 << std::endl;
		distances.push_back(distPointPolyline(p1, l2, seg_ind_2));
		p2 = seg_num_2_vertex(seg_ind_2, l2);
		//std::cout << "p2 = " << p2 << std::endl;
		l_aggreagte.push_back(0.5 * (p1 + p2));
		//std::cout << "dist = " << (p2 -p1).norm() << std::endl;
	}
	Eigen::VectorXd distances_eigen;

	igl::list_to_matrix(distances, distances_eigen);

	dist_average = distances_eigen.mean();
}



double findMaxDim(Eigen::MatrixXd& V1, Eigen::MatrixXd& V2)
{
	Eigen::MatrixXd max_vals = Eigen::MatrixXd::Constant(2, 3, -1);
	Eigen::MatrixXd min_vals = Eigen::MatrixXd::Constant(2, 3, -1);


	max_vals.row(0) = V1.colwise().maxCoeff();
	max_vals.row(1) = V2.colwise().maxCoeff();

	min_vals.row(0) = V1.colwise().minCoeff();
	min_vals.row(1) = V2.colwise().minCoeff();

	Eigen::MatrixXd dimension = (max_vals.colwise().maxCoeff() - min_vals.colwise().minCoeff());

	//double max_dim = (dimension.rowwise().maxCoeff()).norm();
	double max_dim = (dimension.rowwise().minCoeff()).norm();
	return max_dim;
}


void linesToVerticesEdges(const std::vector<Line>& lines, 					
						std::vector<std::vector<int>>& Edges,
					    Eigen::MatrixXd& V)
{
	Edges.clear();
	std::vector<std::vector<double>> Vertices;

	std::vector<double> vertex = { 0., 0., 0 };
	std::vector<int> edge = { 0, 0 };

	for (auto line : lines)
	{

		for (int i = 0; i < line.size(); i++)
		{
			vertex[0] = line[i][0];
			vertex[1] = line[i][1];
			vertex[2] = line[i][2];

			Vertices.push_back(vertex);
			if (i > 0)
			{
				edge[0] = Vertices.size() - 1;
				edge[1] = Vertices.size();
				Edges.push_back(edge);
			}
		}
	}

	
	igl::list_to_matrix(Vertices, V);
}