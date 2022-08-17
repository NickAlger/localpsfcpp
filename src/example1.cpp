#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <Eigen/LU>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <hlib.hh>

#include "lpsf_utils.h"
#include "kdtree.h"
#include "aabbtree.h"
#include "simplexmesh.h"
#include "brent_minimize.h"
#include "ellipsoid.h"
#include "interpolation.h"
#include "impulse_response.h"
#include "product_convolution_kernel.h"
#include "hmatrix.h"

namespace py = pybind11;


int main()
{
    // Options
    int    num_neighbors          = 10;
    double ellipsoid_tau          = 3.0;
    double hmatrix_tol            = 1.0e-6;
    double bct_admissibility_eta  = 2.0;
    int    cluster_size_cutoff    = 32;
    int    min_vol_rtol           = 1e-5;
    int    num_initial_batches    = 0;

    INTERP::ShiftMethod   shift_method   = INTERP::ShiftMethod::ELLIPSOID_MAPPING;
    INTERP::ScalingMethod scaling_method = INTERP::ScalingMethod::DETVOL;

    // std::vector<int> all_num_batches = {1, 5, 25};
    std::vector<int> all_num_batches = {1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100};

    // Load data from file
    Eigen::MatrixXd mesh_vertices        = LPSFUTIL::readMatrix("../data/mesh_vertices.txt").transpose();
    Eigen::MatrixXd mesh_cells_double    = LPSFUTIL::readMatrix("../data/mesh_cells.txt")   .transpose();
    Eigen::MatrixXd dof_coords           = LPSFUTIL::readMatrix("../data/dof_coords.txt")   .transpose();
    Eigen::MatrixXd Hdgn                 = LPSFUTIL::readMatrix("../data/Hdgn_array.txt");
    Eigen::VectorXd vertex_in_dof_double = LPSFUTIL::readMatrix("../data/vertex_in_dof.txt");
    Eigen::VectorXd dof_in_vertex_double = LPSFUTIL::readMatrix("../data/dof_in_vertex.txt");
    Eigen::VectorXd mass_lumps           = LPSFUTIL::readMatrix("../data/mass_matrix_rowsums.txt");

    Eigen::MatrixXi mesh_cells    = LPSFUTIL::matrix_double_to_int(mesh_cells_double);
    Eigen::VectorXi vertex_in_dof = LPSFUTIL::matrix_double_to_int(vertex_in_dof_double);
    Eigen::VectorXi dof_in_vertex = LPSFUTIL::matrix_double_to_int(dof_in_vertex_double);

    std::cout << "mesh_vertices shape=(" << mesh_vertices.rows() << ", " << mesh_vertices.cols() << ")" << std::endl;
    std::cout << "mesh_cells shape=("    << mesh_cells   .rows() << ", " << mesh_cells   .cols() << ")" << std::endl;
    std::cout << "dof_coords shape=("    << dof_coords   .rows() << ", " << dof_coords   .cols() << ")" << std::endl;
    std::cout << "dof_in_vertex shape=(" << dof_in_vertex.rows() << ", " << dof_in_vertex.cols() << ")" << std::endl;
    std::cout << "Hdgn shape=("          << Hdgn         .rows() << ", " << Hdgn         .cols() << ")" << std::endl;
    std::cout << "vertex_in_dof size=" << vertex_in_dof.size() << std::endl;
    std::cout << "dof_in_vertex size=" << dof_in_vertex.size() << std::endl;
    std::cout << "mass_lumps size="    << mass_lumps   .size() << std::endl;

    std::function<Eigen::VectorXd(Eigen::VectorXd)> apply_Hdgn  
        = [&Hdgn](Eigen::VectorXd x) { return Hdgn * x; };

    std::function<Eigen::VectorXd(Eigen::VectorXd)> apply_ML 
        = [&mass_lumps](Eigen::VectorXd x) { return (x.array() * mass_lumps.array()).matrix(); };

    std::function<Eigen::VectorXd(Eigen::VectorXd)> solve_ML 
        = [&mass_lumps](Eigen::VectorXd x) { return (x.array() / mass_lumps.array()).matrix(); };

    Eigen::MatrixXd mesh_vertices_dof_order(mesh_vertices.rows(), mesh_vertices.cols());
    for ( int ii=0; ii<mesh_vertices_dof_order.rows(); ++ii )
    {
        for ( int jj=0; jj<mesh_vertices_dof_order.cols(); ++jj )
        {
            mesh_vertices_dof_order(ii,jj) = mesh_vertices(ii, dof_in_vertex(jj));
        }
    }

    Eigen::MatrixXi mesh_cells_dof_order(mesh_cells.rows(), mesh_cells.cols());
    for ( int ii=0; ii<mesh_cells_dof_order.rows(); ++ii )
    {
        for ( int jj=0; jj<mesh_cells_dof_order.cols(); ++jj )
        {
            mesh_cells_dof_order(ii,jj) = vertex_in_dof(mesh_cells(ii,jj));
        }
    }

    std::vector<Eigen::VectorXd> dof_coords_vector;
    for ( int ii=0; ii<dof_coords.cols(); ++ii )
    {
        dof_coords_vector.push_back(dof_coords.col(ii));
    }

    std::vector<unsigned long int> one_through_N(Hdgn.cols());
    for ( unsigned long int ii=0; ii<Hdgn.cols(); ++ii )
    {
        one_through_N[ii] = ii;
    }

    // Build dense kernel K = diag(1/mass_lumps) * Hdgn *  diag(1/mass_lumps)
    Eigen::MatrixXd K_true(Hdgn.rows(), Hdgn.cols());
    for ( int ii=0; ii<Hdgn.rows(); ++ii )
    {
        for ( int jj=0; jj<Hdgn.cols(); ++jj )
        {
            K_true(ii,jj) = Hdgn(ii,jj) / (mass_lumps(ii) * mass_lumps(jj));
        }
    }

    // Build cluster tree
    std::shared_ptr<HLIB::TClusterTree> ct_ptr 
        = HMAT::build_cluster_tree_from_dof_coords(dof_coords_vector, 32);

    // Build block cluster tree
    std::shared_ptr<HLIB::TBlockClusterTree> bct_ptr
        = HMAT::build_block_cluster_tree(ct_ptr, ct_ptr, bct_admissibility_eta);

    // Create Local PSF Kernel object
    std::shared_ptr<PCK::LPSFKernel> lpsf_kernel_ptr 
        = PCK::create_LPSFKernel(apply_Hdgn, apply_Hdgn, 
                                 apply_ML, apply_ML, 
                                 solve_ML, solve_ML, 
                                 mesh_vertices_dof_order, mesh_vertices_dof_order, 
                                 mesh_cells_dof_order, 
                                 ellipsoid_tau, num_neighbors, min_vol_rtol, num_initial_batches);

    //Add batches to kernel, compute hmatrix, convert hmatrix to dense, check error
    std::vector<double> actual_num_batches;
    std::vector<double> errs;
    for ( int nb : all_num_batches )
    {
        while ( lpsf_kernel_ptr->num_batches() < nb )
        {
            int nb_old = lpsf_kernel_ptr->num_batches();

            lpsf_kernel_ptr->add_batch();

            // It may happen that we run out of points and cannot add another batch
            if ( lpsf_kernel_ptr->num_batches() == nb_old )
            {
                break;
            }
        }
        std::shared_ptr<HLIB::TMatrix> kernel_hmatrix_ptr
            = HMAT::build_lpsfkernel_hmatrix(lpsf_kernel_ptr, bct_ptr, 
                                             shift_method, scaling_method,
                                             hmatrix_tol, true);

        Eigen::MatrixXd K = HMAT::TMatrix_submatrix( kernel_hmatrix_ptr, bct_ptr, 
                                                     one_through_N, one_through_N );
        
        double err = (K_true - K).norm() / K_true.norm();
        std::cout << "num_batches=" << lpsf_kernel_ptr->num_batches() << ", err=" << err << std::endl;
        
        actual_num_batches.push_back(lpsf_kernel_ptr->num_batches());
        errs              .push_back(err);
    }

    // display results
    for ( int ii=0; ii<errs.size(); ++ii )
    {
        std::cout << "num_batches=" << actual_num_batches[ii] << ", err=" << errs[ii] << std::endl;
    }

    return 0;
}