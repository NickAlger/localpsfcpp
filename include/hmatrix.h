#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <hlib.hh>

#include "interpolation.h"
#include "product_convolution_kernel.h"
#include "lpsf_utils.h"


namespace HMAT {

#if HLIB_SINGLE_PREC == 1
using  real_t = float;
#else
using  real_t = double;
#endif


std::shared_ptr<HLIB::TMatrix> build_hmatrix_from_coefffn( HLIB::TCoeffFn<real_t> &                 coefffn,
                                                           std::shared_ptr<HLIB::TBlockClusterTree> bct_ptr,
                                                           double                                   tol,
                                                           bool                                     display )
{
    const HLIB::TClusterTree * row_ct_ptr = bct_ptr->row_ct();
    const HLIB::TClusterTree * col_ct_ptr = bct_ptr->col_ct();

    
    HLIB::TTruncAcc                 acc( tol, 0.0 );
    HLIB::TPermCoeffFn< real_t >    permuted_coefffn( & coefffn, row_ct_ptr->perm_i2e(), col_ct_ptr->perm_i2e() );
    HLIB::TACAPlus< real_t >        aca( & permuted_coefffn );
    HLIB::TDenseMBuilder< real_t >  h_builder( & permuted_coefffn, & aca );
    h_builder.set_coarsening( false );

    std::unique_ptr<HLIB::TMatrix>  A;

    if (display)
    {
        std::cout << "━━ building H-matrix ( tol = " << tol << " )" << std::endl;
        HLIB::TTimer                    timer( HLIB::WALL_TIME );
        HLIB::TConsoleProgressBar       progress;
        timer.start();

        A = h_builder.build( bct_ptr.get(), acc, & progress );

        timer.pause();
        std::cout << "    done in " << timer << std::endl;
        std::cout << "    size of H-matrix = " << HLIB::Mem::to_string( A->byte_size() ) << std::endl;
    }
    else
    {
        A = h_builder.build( bct_ptr.get(), acc, nullptr );
    }

    return std::move(A);
}


// This temporary wrapper is only used to build a Hmatrix from a kernel
struct LPSFKernelHPRO : public HLIB::TCoeffFn< real_t >
{
    std::shared_ptr<PCK::LPSFKernel> Ker_ptr;
    INTERP::ShiftMethod              shift_method;
    INTERP::ScalingMethod            scaling_method;
    INTERP::InterpolationMethod      interpolation_method;

    void eval( const std::vector< HLIB::idx_t > &  rowidxs,
               const std::vector< HLIB::idx_t > &  colidxs,
               real_t *                            matrix ) const
    {
        unsigned long int nrow = rowidxs.size();
        unsigned long int ncol = colidxs.size();

        std::vector<unsigned long int> target_inds(nrow);
        for ( unsigned long int ii=0; ii<nrow; ++ii )
        {
            target_inds[ii] = rowidxs[ii];
        }
        std::vector<unsigned long int> source_inds(ncol);
        for ( unsigned long int jj=0; jj<ncol; ++jj )
        {
            source_inds[jj] = colidxs[jj];
        }

        Eigen::MatrixXd K_block = Ker_ptr->block( target_inds, source_inds, 
                                                  shift_method, scaling_method, interpolation_method );

        for ( size_t  jj = 0; jj < ncol; ++jj )
        {
            for ( size_t  ii = 0; ii < nrow; ++ii )
            {
                matrix[ jj*nrow + ii ] = K_block(ii, jj);
                // matrix[ jj*nrow + ii ] += 1.0e-14; // Used to have some bug where code segfaulted without this
            }
        }
    }

    using HLIB::TCoeffFn< real_t >::eval;
    virtual HLIB::matform_t  matrix_format  () const { return HLIB::MATFORM_NONSYM; }
};

std::shared_ptr<HLIB::TMatrix> build_lpsfkernel_hmatrix(std::shared_ptr<PCK::LPSFKernel>         Ker_ptr,
                                                        std::shared_ptr<HLIB::TBlockClusterTree> bct_ptr,
                                                        INTERP::ShiftMethod                      shift_method,
                                                        INTERP::ScalingMethod                    scaling_method,
                                                        INTERP::InterpolationMethod              interpolation_method,
                                                        double                                   tol,
                                                        bool                                     display)
{
    LPSFKernelHPRO coefffn;
    coefffn.Ker_ptr              = Ker_ptr;
    coefffn.shift_method         = shift_method;
    coefffn.scaling_method       = scaling_method;
    coefffn.interpolation_method = interpolation_method;
    return build_hmatrix_from_coefffn( coefffn, bct_ptr, tol, display );
}

std::shared_ptr<HLIB::TClusterTree> 
    build_cluster_tree_from_dof_coords(const std::vector<Eigen::VectorXd> & dof_coords, double nmin)
{
    size_t N = dof_coords.size();
    size_t d = dof_coords[0].size();

    std::vector< double * >  vertices( N );

    for ( size_t i = 0; i < N; i++ )
    {
        double * v    = new double[d];
        for (size_t j=0; j < d; ++j)
            v[j] = dof_coords[i](j);

        vertices[i] = v;
    }

    auto coord = std::make_unique< HLIB::TCoordinate >( vertices, d );

    HLIB::TCardBSPPartStrat  part_strat;
    HLIB::TBSPCTBuilder      ct_builder( & part_strat, nmin );
    std::unique_ptr<HLIB::TClusterTree>  ct = ct_builder.build( coord.get() );
    return std::move(ct);
}

std::shared_ptr<HLIB::TBlockClusterTree> build_block_cluster_tree(std::shared_ptr<HLIB::TClusterTree> row_ct_ptr,
                                                                  std::shared_ptr<HLIB::TClusterTree> col_ct_ptr,
                                                                  double admissibility_eta)
{
        HLIB::TStdGeomAdmCond    adm_cond( admissibility_eta );
        HLIB::TBCBuilder         bct_builder;
        std::unique_ptr<HLIB::TBlockClusterTree>  bct = bct_builder.build( row_ct_ptr.get(),
                                                                           col_ct_ptr.get(), & adm_cond );
        return std::move(bct);
}

void visualize_cluster_tree(std::shared_ptr<HLIB::TClusterTree> ct_ptr, std::string title)
{
    HLIB::TPSClusterVis c_vis;
    c_vis.print( ct_ptr.get()->root(), title );
}

void visualize_block_cluster_tree(std::shared_ptr<HLIB::TBlockClusterTree> bct_ptr, std::string title)
{
    HLIB::TPSBlockClusterVis bc_vis;
    bc_vis.print( bct_ptr.get()->root(), title );
}

void visualize_hmatrix(std::shared_ptr<HLIB::TMatrix> A_ptr, std::string title)
{
    HLIB::TPSMatrixVis mvis;
    mvis.svd( true );
    mvis.print( A_ptr.get(), title );
}

double TMatrix_entry(const std::shared_ptr<HLIB::TMatrix>           A_ptr, 
                     const std::shared_ptr<HLIB::TBlockClusterTree> bct_ptr,
                     unsigned long int row_ind, unsigned long int col_ind)
{
    return A_ptr->entry(bct_ptr->row_ct()->perm_e2i()->permute(row_ind), 
                        bct_ptr->col_ct()->perm_e2i()->permute(col_ind));
}

Eigen::MatrixXd TMatrix_submatrix( const std::shared_ptr<HLIB::TMatrix>           A_ptr, 
                                   const std::shared_ptr<HLIB::TBlockClusterTree> bct_ptr,
                                   const std::vector<unsigned long int> &         row_inds,
                                   const std::vector<unsigned long int> &         col_inds )
{
    unsigned long int nrow = row_inds.size();
    unsigned long int ncol = col_inds.size();

    Eigen::MatrixXd submatrix(nrow, ncol);
    for ( unsigned long int ii=0; ii<nrow; ++ii )
    {
        for ( unsigned long int jj=0; jj<ncol; ++jj )
        {
            submatrix(ii,jj) = TMatrix_entry(A_ptr, bct_ptr, row_inds[ii], col_inds[jj]);
        }
    }
    return submatrix;
}

Eigen::MatrixXd TMatrix_to_array( const std::shared_ptr<HLIB::TMatrix>           A_ptr, 
                                  const std::shared_ptr<HLIB::TBlockClusterTree> bct_ptr )
{
    std::vector<unsigned long int> row_inds = LPSFUTIL::arange<unsigned long int>(0, A_ptr->rows());
    std::vector<unsigned long int> col_inds = LPSFUTIL::arange<unsigned long int>(0, A_ptr->cols());
    return TMatrix_submatrix(A_ptr, bct_ptr, row_inds, col_inds );
}

}