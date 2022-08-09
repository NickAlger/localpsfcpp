#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include <hlib.hh>
#include <Eigen/LU>

#include "kdtree.h"
#include "simplexmesh.h"
#include "interpolation.h"

namespace PCK {


#if HLIB_SINGLE_PREC == 1
using  real_t = float;
#else
using  real_t = double;
#endif


class ImpulseResponseBatches
{
private:

public:
    int                          dim;
    SMESH::SimplexMesh           mesh;

    std::vector<double>          mesh_vertex_vol;
    std::vector<Eigen::VectorXd> mesh_vertex_mu;
    std::vector<Eigen::MatrixXd> mesh_vertex_Sigma;

    std::vector<Eigen::VectorXd> sample_points;
    std::vector<double>          sample_vol;
    std::vector<Eigen::VectorXd> sample_mu;
    std::vector<Eigen::MatrixXd> sample_Sigma;

    std::vector<Eigen::VectorXd> psi_batches;
    std::vector<int>             point2batch;
    std::vector<int>             batch2point_start;
    std::vector<int>             batch2point_stop;

    double                  tau;
    int                     num_neighbors;
    KDT::KDTree             kdtree;

    ImpulseResponseBatches( const Eigen::Ref<const Eigen::MatrixXd> mesh_vertices, // shape=(dim, num_vertices)
                            const Eigen::Ref<const Eigen::MatrixXi> mesh_cells,    // shape=(dim+1, num_cells)
                            const std::vector<double>               mesh_vertex_vol,
                            const std::vector<Eigen::VectorXd>      mesh_vertex_mu, // size = num_vertices, mesh_vertex_mu[j] has shape (d,)
                            const std::vector<Eigen::MatrixXd>      mesh_vertex_Sigma,
                            int                                     num_neighbors,
                            double                                  tau )
        : mesh(mesh_vertices, mesh_cells), num_neighbors(num_neighbors), tau(tau),
        mesh_vertex_vol(mesh_vertex_vol), mesh_vertex_mu(mesh_vertex_mu), mesh_vertex_Sigma(mesh_vertex_Sigma)
    {
        dim = mesh_vertices.rows();
    }

    void build_kdtree()
    {
        Eigen::MatrixXd pts_matrix(dim, num_pts());
        for ( int ii=0; ii<sample_points.size(); ++ii )
        {
            pts_matrix.col(ii) = sample_points[ii];
        }
        kdtree.build_tree(pts_matrix);
    }

    int num_pts() const
    {
        return sample_points.size();
    }

    int num_batches() const
    {
        return psi_batches.size();
    }

    void add_batch( const Eigen::VectorXi & batch_point_inds, // indices of sample points in the batch we are adding
                    const Eigen::VectorXd & impulse_response_batch, // shape = num_vertices
                    bool                    rebuild_kdtree )
    {
        int num_new_pts = batch_point_inds.size();

        batch2point_start.push_back(num_pts());
        int batch_ind = psi_batches.size();
        for ( int ii=0; ii<num_new_pts; ++ii )
        {
            int ind = batch_point_inds(ii);
            Eigen::VectorXd xi    = mesh.vertices.col(ind);
            double          vol   = mesh_vertex_vol[ind];
            Eigen::VectorXd mu    = mesh_vertex_mu[ind];
            Eigen::MatrixXd Sigma = mesh_vertex_Sigma[ind];

            point2batch  .push_back( batch_ind );
            sample_points.push_back( xi );
            sample_vol   .push_back( vol );
            sample_mu    .push_back( mu );
            sample_Sigma .push_back( Sigma );
        }
        batch2point_stop.push_back(num_pts());

        psi_batches.push_back(impulse_response_batch);

        if ( rebuild_kdtree )
        {
            build_kdtree();
        }
    }

    std::vector<std::pair<Eigen::VectorXd, double>> interpolation_points_and_values(const Eigen::VectorXd & y, // 2d/3d
                                                                                    const Eigen::VectorXd & x, // 2d/3d
                                                                                    const bool mean_shift,
                                                                                    const bool vol_preconditioning) const
    {
        std::pair<Eigen::VectorXi, Eigen::MatrixXd> IC_x = mesh.first_point_collision( x );
        int simplex_ind_x               = IC_x.first(0);
        Eigen::VectorXd affine_coords_x = IC_x.second.col(0);

        double   vol_at_x = 0.0;
        Eigen::VectorXd mu_at_x(dim);
        mu_at_x.setZero();
        if ( simplex_ind_x >= 0 ) // if x is in the mesh
        {
            for ( int kk=0; kk<dim+1; ++kk )
            {
                int vv = mesh.cells(kk, simplex_ind_x);
                vol_at_x += affine_coords_x(kk) * mesh_vertex_vol[vv];
                mu_at_x  += affine_coords_x(kk) * mesh_vertex_mu [vv];
            }
        }

        std::pair<Eigen::VectorXi, Eigen::VectorXd> nn_result = kdtree.query( x, std::min(num_neighbors, num_pts()) );
        Eigen::VectorXi nearest_inds = nn_result.first;

        int N_nearest = nearest_inds.size();

        std::vector<int>             all_simplex_inds (N_nearest);
        std::vector<Eigen::VectorXd> all_affine_coords(N_nearest);
        std::vector<bool>            ind_is_good      (N_nearest);
        for ( int jj=0; jj<N_nearest; ++jj )
        {
            int ind = nearest_inds(jj);
            Eigen::VectorXd xj = sample_points[ind];
            Eigen::VectorXd mu_at_xj = sample_mu[ind];

            Eigen::VectorXd z;
            if (mean_shift)
            {
                z = y - mu_at_x + mu_at_xj;
            }
            else
            {
                z = y - x + xj;
            }

            std::pair<Eigen::VectorXi, Eigen::MatrixXd> IC = mesh.first_point_collision( z );
            all_simplex_inds[jj]  = IC.first(0);
            all_affine_coords[jj] = IC.second.col(0);
            ind_is_good[jj] = ( all_simplex_inds[jj] >= 0 ); // y-x+xi is in mesh => varphi_i(y-x) is defined
        }

        std::vector<std::pair<Eigen::VectorXd, double>> good_points_and_values;
        good_points_and_values.reserve(ind_is_good.size());
        for ( int jj=0; jj<N_nearest; ++jj )
        {
            if ( ind_is_good[jj] )
            {
                int ind = nearest_inds[jj];
                Eigen::VectorXd xj          = sample_points[ind];
                double          vol_at_xj   = sample_vol   [ind];
                Eigen::VectorXd mu_at_xj    = sample_mu    [ind];
                Eigen::MatrixXd Sigma_at_xj = sample_Sigma [ind];

                Eigen::VectorXd dp;
                if (mean_shift)
                {
                    dp = y - mu_at_x;
                }
                else
                {
                    dp = y - x + xj - mu_at_xj;
                }

                double varphi_at_y_minus_x = 0.0;
                if ( dp.transpose() * Sigma_at_xj.ldlt().solve( dp ) < tau*tau )
                {
                    int b = point2batch[ind];
                    const Eigen::VectorXd & phi_j = psi_batches[b];
                    for ( int kk=0; kk<dim+1; ++kk )
                    {
                        if (vol_preconditioning)
                        {
                            varphi_at_y_minus_x += vol_at_x * all_affine_coords[jj](kk) * phi_j(mesh.cells(kk, all_simplex_inds[jj]));
                        }
                        else
                        {
                            varphi_at_y_minus_x += vol_at_xj * all_affine_coords[jj](kk) * phi_j(mesh.cells(kk, all_simplex_inds[jj]));
                        }
                    }
                }
                good_points_and_values.push_back(std::make_pair(xj - x, varphi_at_y_minus_x));
            }
        }
        return good_points_and_values;
    }
};


class ProductConvolutionKernelRBF : public HLIB::TCoeffFn< real_t >
{
private:
    int dim;
    std::shared_ptr<ImpulseResponseBatches> col_batches;

public:
    std::vector<Eigen::VectorXd> row_coords;
    std::vector<Eigen::VectorXd> col_coords;
    bool                         mean_shift = true;
    bool                         vol_preconditioning = true;

    ProductConvolutionKernelRBF( std::shared_ptr<ImpulseResponseBatches> col_batches,
                                         std::vector<Eigen::VectorXd>            col_coords,
                                         std::vector<Eigen::VectorXd>            row_coords)
        : col_batches(col_batches),
          row_coords(row_coords),
          col_coords(col_coords)
    {
        std::cout << "Using ProductConvolutionKernelRBFColsOnly!" << std::endl;
        dim = col_batches->dim;
    }

    double eval_integral_kernel(const Eigen::VectorXd & y,
                                const Eigen::VectorXd & x ) const
    {
        std::vector<std::pair<Eigen::VectorXd, double>> points_and_values;
        if ( col_batches->num_pts() > 0 )
        {
            points_and_values = col_batches->interpolation_points_and_values(y, x, mean_shift, vol_preconditioning); // forward
        }

        int actual_num_pts = points_and_values.size();
        double kernel_value = 0.0;

        int exact_column_index = -1;
        for ( int ii=0; ii<actual_num_pts; ++ii )
        {
            if ( points_and_values[ii].first.norm() < 1e-9 )
            {
                exact_column_index = ii;
                kernel_value = points_and_values[ii].second;
                break;
            }
        }

        if ( exact_column_index < 0 )
        {
            if ( actual_num_pts > 0 )
            {
                Eigen::MatrixXd P(dim, actual_num_pts);
                Eigen::VectorXd F(actual_num_pts);
                for ( int jj=0; jj<actual_num_pts; ++jj )
                {
                    P.col(jj) = points_and_values[jj].first;
                    F(jj)     = points_and_values[jj].second;
                }
                kernel_value = INTERP::tps_interpolate( F, P, Eigen::MatrixXd::Zero(dim,1) );
            }
        }

        return kernel_value;
    }

    double eval_matrix_entry(const int row_ind,
                             const int col_ind ) const
    {
        return eval_integral_kernel(row_coords[row_ind], col_coords[col_ind]);
    }

    void eval  ( const std::vector< HLIB::idx_t > &  rowidxs,
                 const std::vector< HLIB::idx_t > &  colidxs,
                 real_t *                            matrix ) const
    {
        // Check input sizes
        bool input_is_good = true;
        for ( int rr : rowidxs )
        {
            if ( rr < 0 )
            {
                std::string error_message = "Negative row index. rr=";
                error_message += std::to_string(rr);
                throw std::invalid_argument( error_message );
            }
            else if ( rr >= row_coords.size() )
            {
                std::string error_message = "Row index too big. rr=";
                error_message += std::to_string(rr);
                error_message += ", row_coords.size()=";
                error_message += std::to_string(row_coords.size());
                throw std::invalid_argument( error_message );
            }
        }
        for ( int cc : colidxs )
        {
            if ( cc < 0 )
            {
                std::string error_message = "Negative col index. cc=";
                error_message += std::to_string(cc);
                throw std::invalid_argument( error_message );
            }
            else if ( cc >= col_coords.size() )
            {
                std::string error_message = "Col index too big. cc=";
                error_message += std::to_string(cc);
                error_message += ", col_coords.size()=";
                error_message += std::to_string(col_coords.size());
                throw std::invalid_argument( error_message );
            }
        }

        int nrow = rowidxs.size();
        int ncol = colidxs.size();
        for ( size_t  jj = 0; jj < ncol; ++jj )
        {
            for ( size_t  ii = 0; ii < nrow; ++ii )
            {
                matrix[ jj*nrow + ii ] = eval_matrix_entry(rowidxs[ii], colidxs[jj]);
                matrix[ jj*nrow + ii ] += 1.0e-14; // Code segfaults without this
            }
        }

    }

    using HLIB::TCoeffFn< real_t >::eval;

    virtual HLIB::matform_t  matrix_format  () const { return HLIB::MATFORM_NONSYM; }

};

} // end namespace PCK
