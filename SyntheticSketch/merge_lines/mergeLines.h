#ifndef mergeLines_h
#define mergeLines_h

#include "lines.h"
#include <string>

void removeShortLines(std::vector<Line>& lines, std::vector<double>& lines_lengths, const double length_threshold);
double computeLineLength(const Line& line);

void mergeLines(std::vector<Line>& lines,
	Eigen::MatrixXd& V1_,
	Eigen::MatrixXd& V2_,
	double threshold = 0.01);

double findMaxDim(Eigen::MatrixXd& V1, Eigen::MatrixXd& V2);
void findCommonRegion(Line l1, Line l2, std::vector<double>& seg_l1, std::vector<double>& seg_l2, double& dist);
void findCommonRegionClosedOpen(const Line& lclosed, const Line& lopen, std::vector<double>& seg_l1, std::vector<double>& seg_l2);

bool isClosedCurve(Line l);

void findDistancesToAllOtherLines(std::vector<Line>& lines, std::vector<double>& lines_lengths, int ind_l_cur, const double dist_threshold, std::vector<int>& lines_inds, std::vector<double>& distances);
void distancePolylinePolyline(const Line l1, Line& l2, const double dist_threshold, double& distance, bool& aligned);
double distPointPolyline(Eigen::VectorXd p, Line l, double& seg_ind);
double distancePtSegment(Eigen::VectorXd& a, Eigen::VectorXd b, Eigen::VectorXd p, double& projected_point_t);

bool checkAlignement(const Line l1, Line& l2, const int pi, const double seg_temp);
void invertLine(Line& l);
bool reverseTheOrderIfNeeded(Line& l, std::vector<double>& seg_l);

Eigen::VectorXd seg_num_2_vertex(double seg_num, const Line& line);

void mergeCommonRegion(Line& l1,
	Line& l2,
	std::vector<double> seg_l1,
	std::vector<double> seg_l2,
	Line& l_aggreagte,
	double& dist_average);

void linesToVerticesEdges(const std::vector<Line>& lines,
	std::vector<std::vector<int>>& Edges,
	Eigen::MatrixXd& V);

void aggregateCurve(Line& l1,
	Line& l2,
	std::vector<double>& seg_l1,
	std::vector<double>& seg_l2,
	Line& l_aggreagte);

void saveLines(const std::vector<Line>& lines, std::string name);
#endif