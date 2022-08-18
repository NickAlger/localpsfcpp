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
    int    num_neighbors          = 10;     // number of impulses used for interpolation
    double ellipsoid_tau          = 3.0;    // How big the ellipsods are (in standard deviations)
    double hmatrix_tol            = 1.0e-6; // relative tolerance for low rank approximation of matrix blocks
    double bct_admissibility_eta  = 2.0;    // block (A,B) is admissible if min(diam(A), diam(B)) < eta*dist(A, B)
    int    cluster_size_cutoff    = 32;
    int    min_vol_rtol           = 1e-5;   // parameter for throwing away impulses with small volumes
    int    num_initial_batches    = 0;

    INTERP::ShiftMethod         shift_method         = INTERP::ShiftMethod::ELLIPSOID_MAPPING;
    INTERP::ScalingMethod       scaling_method       = INTERP::ScalingMethod::DETVOL;
    INTERP::InterpolationMethod interpolation_method = INTERP::InterpolationMethod::RBF_GAUSS;
    // INTERP::InterpolationMethod interpolation_method = INTERP::InterpolationMethod::RBF_THIN_PLATE_SPLINES;
    bool                        use_symmetry         = false; // use_symmetry=true is not implemented yet

    std::vector<int> all_num_batches = {1, 5, 25, 50, 100};


    // Load data from file
    Eigen::MatrixXd vertices   = LPSFUTIL::readMatrix("../data/mesh_vertices.txt").transpose(); // shape=(d,N)
    Eigen::MatrixXd Hdgn       = LPSFUTIL::readMatrix("../data/Hdgn_array.txt");                // shape=(N,N)
    Eigen::VectorXd mass_lumps = LPSFUTIL::readMatrix("../data/mass_matrix_rowsums.txt");       // size =(N,)

    Eigen::MatrixXd cells_double         = LPSFUTIL::readMatrix("../data/mesh_cells.txt").transpose(); // shape=(d+1,N)
    Eigen::VectorXd dof_in_vertex_double = LPSFUTIL::readMatrix("../data/dof_in_vertex.txt");          // size = (N,)
    Eigen::VectorXd vertex_in_dof_double = LPSFUTIL::readMatrix("../data/vertex_in_dof.txt");          // size = N

    Eigen::MatrixXi cells         = LPSFUTIL::matrix_double_to_int(cells_double);         // shape=(d+1,M)
    Eigen::VectorXi dof_in_vertex = LPSFUTIL::matrix_double_to_int(dof_in_vertex_double); // size =N
    Eigen::VectorXi vertex_in_dof = LPSFUTIL::matrix_double_to_int(vertex_in_dof_double); // size =N

    int d = vertices.rows(); // spatial dimension        (e.g., 1, 2, or 3)
    int N = vertices.cols(); // number of mesh vertices  (e.g., thousands+)
    int M = cells.cols();    // number of mesh triangles (e.g., thousands+)

    std::cout << "vertices shape=("      << vertices     .rows() << ", " << vertices     .cols() << ")" << std::endl;
    std::cout << "cells shape=("         << cells        .rows() << ", " << cells        .cols() << ")" << std::endl;
    std::cout << "dof_in_vertex shape=(" << dof_in_vertex.rows() << ", " << dof_in_vertex.cols() << ")" << std::endl;
    std::cout << "Hdgn shape=("          << Hdgn         .rows() << ", " << Hdgn         .cols() << ")" << std::endl;

    std::cout << "dof_in_vertex size=" << dof_in_vertex.size() << std::endl;
    std::cout << "mass_lumps size="    << mass_lumps   .size() << std::endl;


    // Create linear operators that apply the Gauss-Newton Hessian and the lumped mass matrix to vectors
    std::function<Eigen::VectorXd(Eigen::VectorXd)> apply_Hdgn  
        = [&Hdgn](Eigen::VectorXd x) { return Hdgn * x; };

    std::function<Eigen::VectorXd(Eigen::VectorXd)> apply_ML 
        = [&mass_lumps](Eigen::VectorXd x) { return (x.array() * mass_lumps.array()).matrix(); };

    std::function<Eigen::VectorXd(Eigen::VectorXd)> solve_ML 
        = [&mass_lumps](Eigen::VectorXd x) { return (x.array() / mass_lumps.array()).matrix(); };


    // Reorder vertices and cells to the degree of freedom ordering
    Eigen::MatrixXd vertices_dof_order(d, N);
    for ( int ii=0; ii<N; ++ii )
    {
        vertices_dof_order.col(ii) = vertices.col(dof_in_vertex(ii));
    }

    Eigen::MatrixXi cells_dof_order(d+1, M);
    for ( int ii=0; ii<d+1; ++ii )
    {
        for ( int jj=0; jj<M; ++jj )
        {
            cells_dof_order(ii,jj) = vertex_in_dof(cells(ii,jj));
        }
    }

    // unpack vertices from a MatrixXd into a vector of VectorXd. size=N, elm_size=d
    std::vector<Eigen::VectorXd> vertices_dof_order_vector = LPSFUTIL::unpack_MatrixXd_columns(vertices_dof_order);


    // Build dense kernel K = diag(1/mass_lumps) * Hdgn *  diag(1/mass_lumps)
    Eigen::MatrixXd K_true(N, N);
    for ( int ii=0; ii<N; ++ii )
    {
        for ( int jj=0; jj<N; ++jj )
        {
            K_true(ii,jj) = Hdgn(ii,jj) / (mass_lumps(ii) * mass_lumps(jj));
        }
    }

    // Build cluster tree
    std::shared_ptr<HLIB::TClusterTree> ct_ptr 
        = HMAT::build_cluster_tree_from_dof_coords(vertices_dof_order_vector, cluster_size_cutoff);

    // Build block cluster tree
    std::shared_ptr<HLIB::TBlockClusterTree> bct_ptr
        = HMAT::build_block_cluster_tree(ct_ptr, ct_ptr, bct_admissibility_eta);

    // Create Local PSF Kernel object
    std::shared_ptr<PCK::LPSFKernel> lpsf_kernel_ptr 
        = PCK::create_LPSFKernel(apply_Hdgn, apply_Hdgn, 
                                 apply_ML, apply_ML, 
                                 solve_ML, solve_ML, 
                                 vertices_dof_order, vertices_dof_order, 
                                 cells_dof_order, 
                                 ellipsoid_tau, num_neighbors, min_vol_rtol, num_initial_batches);

    //Add batches to kernel, compute hmatrix, convert hmatrix to dense, check error
    std::vector<double> actual_num_batches;
    std::vector<double> errs;
    for ( int nb : all_num_batches )
    {
        // Add batches until the desired number is reached
        while ( lpsf_kernel_ptr->num_batches() < nb )
        {
            bool batch_added_successfully = lpsf_kernel_ptr->add_batch();
            if ( !batch_added_successfully ) // If we run out of points we cannot add another batch
            {
                break;
            }
        }

        // Construct hmatrix
        bool display_progress = true;
        std::shared_ptr<HLIB::TMatrix> kernel_hmatrix_ptr
            = HMAT::build_lpsfkernel_hmatrix(lpsf_kernel_ptr, bct_ptr, 
                                             shift_method, scaling_method, interpolation_method, use_symmetry,
                                             hmatrix_tol, display_progress);

        // Convert hmatrix to dense array to check error (not scalable)
        Eigen::MatrixXd K = HMAT::TMatrix_to_array( kernel_hmatrix_ptr, bct_ptr );
        
        // Compute relative error in lpsf/hmatrix approximation
        double err = (K_true - K).norm() / K_true.norm();
        std::cout << "num_batches=" << lpsf_kernel_ptr->num_batches() << ", err=" << err << std::endl;
        
        actual_num_batches.push_back(lpsf_kernel_ptr->num_batches());
        errs              .push_back(err);
    }

    // display results
    std::cout << std::endl;
    for ( unsigned int ii=0; ii<errs.size(); ++ii )
    {
        std::cout << "num_batches=" << actual_num_batches[ii] << ", err=" << errs[ii] << std::endl;
    }

    return 0;
}