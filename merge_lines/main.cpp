#include "input.h"
#include "readEdges.h"
#include "mergeLines.h"
#include "lines.h"
#include <igl/remove_duplicate_vertices.h>

#include <iostream>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>

#include <igl/png/writePNG.h>
#include <igl/png/readPNG.h>

#include <igl/remove_duplicate_vertices.h>

#include <cstdlib>


using namespace std;
using namespace igl;
using namespace Eigen;
using namespace igl::opengl;

string obj_filename;


void scaleCenterShape(Eigen::MatrixXd& V1, Eigen::MatrixXd& V2)
{
	Eigen::MatrixXd max_vals = Eigen::MatrixXd::Constant(2, 3, -1);		
	Eigen::MatrixXd min_vals = Eigen::MatrixXd::Constant(2, 3, -1);


	max_vals.row(0) = V1.colwise().maxCoeff();
	max_vals.row(1) = V2.colwise().maxCoeff();

	min_vals.row(0) = V1.colwise().minCoeff();
	min_vals.row(1) = V2.colwise().minCoeff();
	
	Eigen::MatrixXd dimension = (max_vals.colwise().maxCoeff() - min_vals.colwise().minCoeff());
	cout << dimension << endl;

	cout << dimension.rowwise().maxCoeff() << endl;
	double object_scale = (dimension.rowwise().maxCoeff()).norm();

	cout << object_scale << endl;

	V1 = V1 / object_scale;
	V2 = V2 / object_scale;

	max_vals.row(0) = V1.colwise().maxCoeff();
	max_vals.row(0) = V2.colwise().maxCoeff();

	min_vals.row(0) = V1.colwise().minCoeff();
	min_vals.row(0) = V2.colwise().minCoeff();

	
	auto centroid = (0.5 * (max_vals.colwise().maxCoeff() + min_vals.colwise().minCoeff())).eval();

	//cout << centroid << endl;
	
	V1 = V1.rowwise() - centroid;
	V2 = V2.rowwise() - centroid;
}

bool to_bool(std::string str) {
	std::transform(str.begin(), str.end(), str.begin(), ::tolower);
	std::istringstream is(str);
	bool b;
	is >> std::boolalpha >> b;
	return b;
}

int to_int(std::string str) {	
	int b;
	b = std::stoi(str);
	return b;
}

double to_double(std::string str) {
	cout << str << endl;
	double b;
	b = std::stod(str);
	cout << b << endl;
	return b;
}

int main(int argc, char* argv[])
{
	InputParser input(argc, argv);



	/*Default values:*/
	string folder_in = "C://Users//yulia//Research//Ling//3d_sketch//add_miss_edges//";
	
	string model_name = "table_0361_opt_quad_network_20_revise.obj";

	//string folder_in = "C://Users//yulia//Research//Ling//3d_sketch//stage2_network//";
	//string model_name = "chair_0001_opt_quad_network_20.obj";

	/*string folder_in = "C://Users//yulia//Research//Ling//3d_sketch//stage2 network//3//";
	string model_name = "2df0d24befaef397549f05ce44760eca_opt_quad_network_20.obj";*/

	//string folder_in = "C://Users//yulia//Research//Ling//3d_sketch//stage2_network//7//";
	//string model_name = "296c92e4e7233de47d1dd50d46b1e3d1_opt_quad_network_20.obj";

	//string folder_in = "C://Users//yulia//Research//Ling//3d_sketch//stage2_network//10//";
	/*string model_name = "296c315f8f0c7d5d87c63d8b3018b58_opt_quad_network_20.obj";*/
	//string model_name = "bed_0426_opt_quad_network_20.obj";


	//string folder_out = "..//results//";

	



	/*Command line values:*/
	if (input.cmdOptionExists("-in"))
		folder_in = input.getCmdOption("-in");

	//if (input.cmdOptionExists("-folder_out"))
	//	folder_out = input.getCmdOption("-folder_out");


	if (input.cmdOptionExists("-model_name"))
		model_name = input.getCmdOption("-model_name");


	/*Filepaths:*/
	obj_filename = folder_in + model_name;

	/*Load mesh:*/
	Eigen::MatrixXd V1; //vertices
	Eigen::MatrixXd V2; //vertices

	std::vector<Line> lines;

	readEdges(obj_filename, V1, V2, lines);

	//saveLines(lines, "original");

	mergeLines(lines, V1, V2, 0.05);


	/*Write lines to an output file*/
	saveLines(lines, "aggredated");
	
}