#include "readEdges.h"

#include <sstream>
#include <iostream>
#include <Eigen/Core>
#include <vector>
#include <iterator>
#include <igl/list_to_matrix.h>

#include "lines.h"

bool readEdges(const std::string obj_file_name,
    Eigen::MatrixXd& V1_,
    Eigen::MatrixXd& V2_,
    std::vector<Line>& lines)
{
    FILE* obj_file = fopen(obj_file_name.c_str(), "r");
    if (NULL == obj_file)
    {
        fprintf(stderr, "IOError: %s could not be opened...\n",
            obj_file_name.c_str());
        return false;
    }

    std::string v("v");
    std::string l_str("l");
    std::string chain_str("#");

    #ifndef IGL_LINE_MAX
    #  define IGL_LINE_MAX 2048
    #endif

    std::vector<std::vector<double> > V;
    std::vector<std::vector<int>> edges;
    std::vector<std::vector<double> > V1;
    std::vector<std::vector<double> > V2;

    int v1, v2, v_last;

    char line[IGL_LINE_MAX];
    int line_no = 1;


    while (fgets(line, IGL_LINE_MAX, obj_file) != NULL)
    {
        char type[IGL_LINE_MAX];
        // Read first word containing type
        if (sscanf(line, "%s", type) == 1)
        {                     
            // Get pointer to rest of line right after type
            
            if (type == v)
            {
                std::istringstream ls(&line[1]);
                std::vector<double > vertex{ std::istream_iterator<double >(ls), std::istream_iterator<double >() };

                if (vertex.size() < 3)
                {
                    fprintf(stderr,
                        "Error: readEdges() vertex on line %d should have at least 3 coordinates",
                        line_no);
                    fclose(obj_file);
                    return false;
                }

                V.push_back(vertex);
            }
            else if (type == l_str)
            {
                std::istringstream ls(&line[1]);
                std::vector<int > edge{ std::istream_iterator<int >(ls), std::istream_iterator<int >() };

                if (edge.size() < 2)
                {
                    fprintf(stderr,
                        "Error: readEdges() edge %d should have at least 2 coordinates",
                        line_no);
                    fclose(obj_file);
                    return false;
                }

                V1.push_back(V[edge[0]-1]);
                V2.push_back(V[edge[1]-1]);
                edges.push_back(edge);
            }
            else if (type == chain_str)
            {
                char* l = &line[strlen(type)+1];
                sscanf(l, "%s", type);
                
                l = &l[strlen(type)];
                
                std::istringstream ls(&l[1]);

                std::vector<int> edges_in_line{ std::istream_iterator<int >(ls), std::istream_iterator<int >() };

                //Get the line
                Eigen::VectorXd v(3);
                int e_ind;
                Line line_obj;
                
                //std::cout << "---------------------------------------" << std::endl;

                for (int ej = 0; ej < edges_in_line.size(); ej++)
                {
                    e_ind = edges_in_line[ej]-1;

                    v1 = edges[e_ind][0];
                    v2 = edges[e_ind][1];

                    /*std::cout << e_ind << std::endl;
                    std::cout << V1[e_ind][0] << std::endl;
                    std::cout << V1[e_ind][1] << std::endl;*/

             /*       v[0] = V1[e_ind][0];
                    v[1] = V1[e_ind][1];
                    v[2] = V1[e_ind][2];*/

                    //std::cout << "v = " << v << std::endl;

                    if (line_obj.size() > 1)
                    {
                        /*std::cout << "*line_obj.end() -1 = " << *(line_obj.end() - 1) << std::endl;
                        std::cout << "*line_obj.end() -2 = " << *(line_obj.end() - 2) << std::endl;*/
                        if (v1 != v_last)
                        {
                            int temp = v1;
                            v1 = v2;
                            v2 = temp;
                        }
                        
                        v[0] = V[v1 - 1][0];
                        v[1] = V[v1 - 1][1];
                        v[2] = V[v1 - 1][2];
                        v_last = v2;

                        Eigen::VectorXd edge1 = (v - *(line_obj.end() - 1));
                        Eigen::VectorXd edge2 = (*(line_obj.end()-2) - *(line_obj.end()-1));

                        edge1 = edge1 / edge1.norm();
                        edge2 = edge2 / edge2.norm();

                        /*std::cout << "edge1 " << edge1 << std::endl;
                        std::cout << "edge2 " << edge2 << std::endl;*/

                        if (edge1.dot(edge2) > -0.7) //cosd(135) -- angle samller than 135
                        {
                            //std::cout << abs(edge1.dot(edge2)) << std::endl;
                            lines.push_back(line_obj);
                            Eigen::VectorXd v_ = *(line_obj.end()-1);
                            line_obj.clear();
                            line_obj.push_back(v_);
                        }
                    }
                    else
                    {
                        v_last = v2;
                        v[0] = V[v1-1][0];
                        v[1] = V[v1-1][1];
                        v[2] = V[v1-1][2];
                    }

                    line_obj.push_back(v);
                }
                //std::cout << e_ind << std::endl;
                v[0] = V2[e_ind][0];
                v[1] = V2[e_ind][1];
                v[2] = V2[e_ind][2];
                line_obj.push_back(v);

                lines.push_back(line_obj);
            }
            else
            {
                std::cout << (type == l_str) << std::endl;

                //ignore any other lines
                fprintf(stderr,
                    "Warning: readEdges() ignored non-comment line %d:\n  %s",
                    line_no,
                    line);
            }
        }
        else
        {
            // ignore empty line
        }
        line_no++;
    }
    fclose(obj_file);


    bool V1_rect = igl::list_to_matrix(V1, V1_);
    if (!V1_rect)
    {
        // igl::list_to_matrix(vV,V) already printed error message to std err
        return false;
    }

    bool V2_rect = igl::list_to_matrix(V2, V2_);
    if (!V2_rect)
    {
        // igl::list_to_matrix(vV,V) already printed error message to std err
        return false;
    }

    return true;
    
}
