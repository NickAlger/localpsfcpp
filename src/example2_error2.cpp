#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

// #include <pybind11/pybind11.h>
// #include <pybind11.h>
// #include "pybind11/pybind11.h"
// #include "pybind11.h"

#include <Eigen/Dense>
#include <Eigen/LU>
// #include <eigen3/Eigen/Dense>
// #include <eigen3/Eigen/LU>
// #include <pybind11/eigen.h>
// #include <pybind11/stl.h>
// #include <pybind11/functional.h>

#include <hlib.hh>
#include <hpro/algebra/mat_norm.hh>

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

// namespace py = pybind11;

#if HLIB_SINGLE_PREC == 1
using  real_t = float;
#else
using  real_t = double;
#endif


struct GaussKernel : public HLIB::TCoeffFn< real_t >
{
    Eigen::MatrixXd vertices;
    double sigma_squared;

    GaussKernel( Eigen::MatrixXd vertices_input, double sigma ) 
        : vertices(vertices_input), sigma_squared(sigma*sigma) {}

    void eval( const std::vector< HLIB::idx_t > &  rowidxs,
               const std::vector< HLIB::idx_t > &  colidxs,
               real_t *                            matrix ) const
    {
        unsigned long int nrow = rowidxs.size();
        unsigned long int ncol = colidxs.size();

        for ( int ii=0; ii<nrow; ++ii )
        {
            Eigen::VectorXd p = vertices.col(rowidxs[ii]);
            for ( int jj=0; jj<ncol; ++jj )
            {
                Eigen::VectorXd q = vertices.col(colidxs[jj]);
                double r_squared = (p-q).squaredNorm();
                matrix[ jj*nrow + ii ] = exp(-0.5 * r_squared / sigma_squared);
            }
        }
    }

    using HLIB::TCoeffFn< real_t >::eval;
    virtual HLIB::matform_t  matrix_format  () const { return HLIB::MATFORM_NONSYM; }
};


int main()
{
    // Options
    int    num_neighbors          = 10;     // number of impulses used for interpolation
    double hmatrix_tol            = 1.0e-9; // relative tolerance for low rank approximation of matrix blocks
    double bct_admissibility_eta  = 2.0;    // block (A,B) is admissible if min(diam(A), diam(B)) < eta*dist(A, B)
    int    cluster_size_cutoff    = 32;
    int    min_vol_rtol           = 1e-5;   // parameter for throwing away impulses with small volumes
    int    num_initial_batches    = 0;

    // INTERP::ShiftMethod         shift_method         = INTERP::ShiftMethod::ELLIPSOID_MAPPING;
    INTERP::ShiftMethod         shift_method         = INTERP::ShiftMethod::LOCAL_MEAN_DISPLACEMENT_INVARIANCE;
    // INTERP::ShiftMethod         shift_method         = INTERP::ShiftMethod::LOCAL_TRANSLATION_INVARIANCE;
    // INTERP::ScalingMethod       scaling_method       = INTERP::ScalingMethod::DETVOL;
    INTERP::ScalingMethod       scaling_method       = INTERP::ScalingMethod::VOL;
    INTERP::InterpolationMethod interpolation_method = INTERP::InterpolationMethod::RBF_GAUSS;
    // INTERP::InterpolationMethod interpolation_method = INTERP::InterpolationMethod::RBF_THIN_PLATE_SPLINES;
    bool                        use_symmetry         = false; // use_symmetry=true is not implemented yet

    double sigma = 0.05;
    double tight_hmatrix_tol = 1e-12;

    std::vector<int>    all_nn          = {64}; //{8, 16, 32, 64}; //, 128}; // mesh is n x n
    std::vector<double> all_tau         = {3.0}; //{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}; // Ellipsoid size in standard deviations
    std::vector<int>    all_num_batches = {32}; //{1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    std::vector<int>    nns;
    std::vector<double> hs;
    std::vector<double> taus;
    std::vector<double> errs;
    std::vector<int>    nbs;
    for ( int n : all_nn)
    {
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> mesh = LPSFUTIL::make_unit_square_mesh(n, n);
        Eigen::MatrixXd vertices = std::get<0>(mesh);
        Eigen::MatrixXi cells    = std::get<1>(mesh);

        std::vector<Eigen::VectorXd> vertices_list = LPSFUTIL::unpack_MatrixXd_columns(vertices);

        // Build cluster tree
        std::shared_ptr<HLIB::TClusterTree> ct_ptr 
            = HMAT::build_cluster_tree_from_dof_coords(vertices_list, cluster_size_cutoff);

        // Build block cluster tree
        std::shared_ptr<HLIB::TBlockClusterTree> bct_ptr
            = HMAT::build_block_cluster_tree(ct_ptr, ct_ptr, bct_admissibility_eta);

        // Create Gaussian kernel hmatrix, accurate to tight tolerance
        GaussKernel GK(vertices, sigma);
        std::shared_ptr<HLIB::TMatrix> Ktrue_ptr = HMAT::build_hmatrix_from_coefffn( GK, 
                                                                                     bct_ptr, 
                                                                                     tight_hmatrix_tol, 
                                                                                     true );

        int N = (n+1)*(n+1);
        double h = 1.0 / ((double)n);
        Eigen::VectorXd mass_lumps = Eigen::VectorXd::Ones(N) / (h*h);

        std::function<Eigen::VectorXd(Eigen::VectorXd)> apply_ML 
            = [&mass_lumps](Eigen::VectorXd x) { return (x.array() * mass_lumps.array()).matrix(); };

        std::function<Eigen::VectorXd(Eigen::VectorXd)> solve_ML 
            = [&mass_lumps](Eigen::VectorXd x) { return (x.array() / mass_lumps.array()).matrix(); };

        std::function<Eigen::VectorXd(Eigen::VectorXd)> apply_A  
            = [&](Eigen::VectorXd x) 
                { return apply_ML(HMAT::TMatrix_matvec(Ktrue_ptr, bct_ptr, apply_ML(x), HLIB::apply_normal)); };

        std::function<Eigen::VectorXd(Eigen::VectorXd)> apply_AT
            = [&](Eigen::VectorXd x) 
                { return apply_ML(HMAT::TMatrix_matvec(Ktrue_ptr, bct_ptr, apply_ML(x), HLIB::apply_transposed)); };


        for ( double tau : all_tau )
        {
            // Create Local PSF Kernel object
            std::shared_ptr<PCK::LPSFKernel> lpsf_kernel_ptr 
                = PCK::create_LPSFKernel(apply_A, apply_AT, 
                                         apply_ML, apply_ML, 
                                         solve_ML, solve_ML, 
                                         vertices, vertices, 
                                         cells, tau, num_neighbors, 
                                         min_vol_rtol, num_initial_batches);

            //Add batches to kernel, compute hmatrix, convert hmatrix to dense, check error
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
                int true_nb = lpsf_kernel_ptr->num_batches();

                std::cout << "n=" << n << ", h=" << h << ", tau=" << tau << ", nb=" << true_nb << std::endl;
                Construct hmatrix
                bool display_progress = true;
                std::shared_ptr<HLIB::TMatrix> K_ptr
                    = HMAT::build_lpsfkernel_hmatrix(lpsf_kernel_ptr, bct_ptr, 
                                                     shift_method, scaling_method, interpolation_method, use_symmetry,
                                                     hmatrix_tol, display_progress);

                double err = HLIB::diff_norm_F(Ktrue_ptr.get(), K_ptr.get(), true);

                std::cout << "n=" << n << ", h=" << h << ", tau=" << tau << ", nb=" << true_nb << ", err=" << err << std::endl;
                
                nns.push_back(n);
                hs.push_back(h);
                taus.push_back(tau);
                nbs.push_back(true_nb);
                errs.push_back(err);
            }
        }
    }

    display results
    std::cout << std::endl;
    for ( unsigned int ii=0; ii<errs.size(); ++ii )
    {
        std::cout << "h=" << hs[ii] << ", tau=" << taus[ii] << ", nb=" << nbs[ii] << ", err=" << errs[ii] << std::endl;
    }

    return 0;
}

// (fenics4) nick@nick-HP-Laptop-17-ca1xxx:~/repos/localpsfcpp/bin$ ./example2
// ━━ building H-matrix ( tol = 1e-12 )
//     done in 1.36s                                                
//     size of H-matrix = 68.87 MB
// computing V                                                      
// computing mu
// ii = 0
// ii = 1
// computing Sigma
// ii = 0, jj = 0
// ii = 1, jj = 0
// ii = 1, jj = 1
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// n=64, h=0.015625, tau=3, nb=32
// ━━ building H-matrix ( tol = 1e-09 )
// ██████████████████████████████████▎       86% ETA 1 s (136.32 MB)corrupted size vs. prev_size
// Aborted (core dumped)




// ---------------------------------------------------------------------------------------------




// (fenics4) nick@nick-HP-Laptop-17-ca1xxx:~/repos/localpsfcpp/bin$ ./example2
// ━━ building H-matrix ( tol = 1e-12 )
//     done in 0.02s                                              
//     size of H-matrix = 57.24 kB
// computing V                                                    
// computing mu
// ii = 0
// ii = 1
// computing Sigma
// ii = 0, jj = 0
// ii = 1, jj = 0
// ii = 1, jj = 1
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// n=8, h=0.125, tau=4, nb=5
// ━━ building H-matrix ( tol = 1e-09 )
//     done in 0.13s                                               
//     size of H-matrix = 57.24 kB
// n=8, h=0.125, tau=4, nb=5, err=0.0359936                        
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// n=8, h=0.125, tau=4, nb=10
// ━━ building H-matrix ( tol = 1e-09 )
//     done in 0.13s                                               
//     size of H-matrix = 57.24 kB
// n=8, h=0.125, tau=4, nb=10, err=0.0321111                       
// n=8, h=0.125, tau=4, nb=10
// ━━ building H-matrix ( tol = 1e-09 )
//     done in 0.14s                                               
//     size of H-matrix = 57.24 kB
// n=8, h=0.125, tau=4, nb=10, err=0.0321111                       
// n=8, h=0.125, tau=4, nb=10
// ━━ building H-matrix ( tol = 1e-09 )
//     done in 0.15s                                               
//     size of H-matrix = 57.24 kB
// n=8, h=0.125, tau=4, nb=10, err=0.0321111                       
// ━━ building H-matrix ( tol = 1e-12 )
//     done in 0.37s                                               
//     size of H-matrix = 487.63 kB
// computing V                                                     
// computing mu
// ii = 0
// ii = 1
// computing Sigma
// ii = 0, jj = 0
// ii = 1, jj = 0
// ii = 1, jj = 1
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// n=16, h=0.0625, tau=4, nb=5
// ━━ building H-matrix ( tol = 1e-09 )
//     done in 0.43s                                               
//     size of H-matrix = 310.97 kB
// n=16, h=0.0625, tau=4, nb=5, err=0.0406423                      
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// n=16, h=0.0625, tau=4, nb=20
// ━━ building H-matrix ( tol = 1e-09 )
//     done in 0.47s                                               
//     size of H-matrix = 310.97 kB
// n=16, h=0.0625, tau=4, nb=20, err=0.0265405                     
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// n=16, h=0.0625, tau=4, nb=51
// ━━ building H-matrix ( tol = 1e-09 )
//     done in 1.22s                                               
//     size of H-matrix = 310.97 kB
// n=16, h=0.0625, tau=4, nb=51, err=0.000525285                   
// n=16, h=0.0625, tau=4, nb=51
// ━━ building H-matrix ( tol = 1e-09 )
//     done in 4.87s                                               
//     size of H-matrix = 310.97 kB
// n=16, h=0.0625, tau=4, nb=51, err=0.000525285                   
// ━━ building H-matrix ( tol = 1e-12 )
//     done in 2.29s                                               
//     size of H-matrix = 5.82 MB
// computing V                                                     
// computing mu
// ii = 0
// ii = 1
// computing Sigma
// ii = 0, jj = 0
// ii = 1, jj = 0
// ii = 1, jj = 1
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// n=32, h=0.03125, tau=4, nb=5
// ━━ building H-matrix ( tol = 1e-09 )
//     done in 1.76s                                               
//     size of H-matrix = 2.04 MB
// n=32, h=0.03125, tau=4, nb=5, err=0.0583743                     
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// n=32, h=0.03125, tau=4, nb=20
// ━━ building H-matrix ( tol = 1e-09 )
//     done in 1.93s                                               
//     size of H-matrix = 2.03 MB
// n=32, h=0.03125, tau=4, nb=20, err=0.0385927                    
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// n=32, h=0.03125, tau=4, nb=100
// ━━ building H-matrix ( tol = 1e-09 )
//     done in 2.02s                                               
//     size of H-matrix = 2.01 MB
// n=32, h=0.03125, tau=4, nb=100, err=0.0170762                   
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// n=32, h=0.03125, tau=4, nb=190
// ━━ building H-matrix ( tol = 1e-09 )
//     done in 9.42s                                               
//     size of H-matrix = 2.00 MB
// n=32, h=0.03125, tau=4, nb=190, err=0.0015069                   

// h=0.125, tau=4, nb=5, err=0.0359936
// h=0.125, tau=4, nb=10, err=0.0321111
// h=0.125, tau=4, nb=10, err=0.0321111
// h=0.125, tau=4, nb=10, err=0.0321111
// h=0.0625, tau=4, nb=5, err=0.0406423
// h=0.0625, tau=4, nb=20, err=0.0265405
// h=0.0625, tau=4, nb=51, err=0.000525285
// h=0.0625, tau=4, nb=51, err=0.000525285
// h=0.03125, tau=4, nb=5, err=0.0583743
// h=0.03125, tau=4, nb=20, err=0.0385927
// h=0.03125, tau=4, nb=100, err=0.0170762
// h=0.03125, tau=4, nb=190, err=0.0015069
// (fenics4) nick@nick-HP-Laptop-17-ca1xxx:~/repos/localpsfcpp/bin$ ./example2
// ━━ building H-matrix ( tol = 1e-12 )
//     done in 1.56s                                                
//     size of H-matrix = 68.87 MB
// computing V                                                      
// computing mu
// ii = 0
// ii = 1
// computing Sigma
// ii = 0, jj = 0
// ii = 1, jj = 0
// ii = 1, jj = 1
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// computing impulse response batch
// n=64, h=0.015625, tau=3, nb=32
// ━━ building H-matrix ( tol = 1e-09 )
// ██████████████████████████████████████▎   96% ETA 0 s (139.73 MB)free(): invalid next size (fast)
// Aborted (core dumped)
