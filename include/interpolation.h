#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include <hlib.hh>
#include <Eigen/LU>

#include "kdtree.h"
#include "simplexmesh.h"


namespace INTERP
{

// Radial basis function interpolation with thin plate spline basis functions
double tps_interpolate( const Eigen::VectorXd & function_at_rbf_points,
                        const Eigen::MatrixXd & rbf_points,
                        const Eigen::VectorXd & eval_point )
{
    int N = rbf_points.cols();

    double function_at_eval_point;
    if ( N == 1 )
    {
        function_at_eval_point = function_at_rbf_points(0);
    }
    else
    {
        Eigen::MatrixXd M(N, N);
        for ( int jj=0; jj<N; ++jj )
        {
            for ( int ii=0; ii<N; ++ii )
            {
                if ( ii == jj )
                {
                    M(ii,jj) = 0.0;
                }
                else
                {
                    double r_squared = (rbf_points.col(ii) - rbf_points.col(jj)).squaredNorm();
                    M(ii, jj) = 0.5 * r_squared * log(r_squared);
                }
            }
        }

        Eigen::VectorXd weights = M.lu().solve(function_at_rbf_points);

        Eigen::VectorXd rbfs_at_eval_point(N);
        for ( int ii=0; ii<N; ++ii )
        {
            double r_squared = (rbf_points.col(ii) - eval_point).squaredNorm();
            if ( r_squared == 0.0 )
            {
                rbfs_at_eval_point(ii) = 0.0;
            }
            else
            {
                rbfs_at_eval_point(ii) = 0.5 * r_squared * log(r_squared);
            }
        }

        function_at_eval_point = (weights.array() * rbfs_at_eval_point.array()).sum();
    }
    return function_at_eval_point;
}


// Local local mean displacement invariance points and values
std::vector<std::pair<Eigen::VectorXd, double>> 
    lmdi_points_and_values(const Eigen::VectorXd              & y, // 2d/3d
                           const Eigen::VectorXd              & x, // 2d/3d
                           const std::vector<Eigen::VectorXd> & impulse_response_batches,
                           const std::vector<int>             & point2batch,
                           const SMESH::SimplexMesh           & mesh,
                           const std::vector<double>          & mesh_vertex_vol,
                           const std::vector<Eigen::VectorXd> & mesh_vertex_mu,
                           const std::vector<Eigen::MatrixXd> & mesh_vertex_Sigma,
                           const std::vector<Eigen::VectorXd> & sample_points,
                           const std::vector<double>          & sample_vol,
                           const std::vector<Eigen::VectorXd> & sample_mu,
                           const std::vector<Eigen::MatrixXd> & sample_Sigma,
                           const double                       & tau,
                           const int                          & num_neighbors,
                           const KDT::KDTree                  & sample_points_kdtree)
{
    const int dim     = y.size();

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

    std::pair<Eigen::VectorXi, Eigen::VectorXd> nn_result = 
        sample_points_kdtree.query( x, std::min(num_neighbors, sample_points.size()) );
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
        z = y - mu_at_x + mu_at_xj;

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
            dp = y - mu_at_x;


            double varphi_at_y_minus_x = 0.0;
            if ( dp.transpose() * Sigma_at_xj.ldlt().solve( dp ) < tau*tau )
            {
                int b = point2batch[ind];
                const Eigen::VectorXd & phi_j = impulse_response_batches[b];
                for ( int kk=0; kk<dim+1; ++kk )
                {
                    varphi_at_y_minus_x += vol_at_x * all_affine_coords[jj](kk) * phi_j(mesh.cells(kk, all_simplex_inds[jj]));
                }
            }
            good_points_and_values.push_back(std::make_pair(xj - x, varphi_at_y_minus_x));
        }
    }
    return good_points_and_values;
}

}