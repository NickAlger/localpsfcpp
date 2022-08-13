#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <algorithm>

#include "kdtree.h"
#include "simplexmesh.h"
#include "ellipsoid.h"


namespace INTERP
{

// Radial basis function interpolation with thin plate spline basis functions
double TPS_interpolate( const Eigen::VectorXd & function_at_rbf_points,
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
    LMDI_points_and_values(unsigned long int                        target_ind,
                           unsigned long int                        source_ind,
                           const std::vector<Eigen::VectorXd>     & source_coords, // size=NS, elm_size=dS
                           const std::vector<Eigen::VectorXd>     & target_coords, // size=NT, elm_size=dT
                           const SMESH::SimplexMesh               & target_mesh,
                           const std::vector<double>              & vol,           // size=NS
                           const std::vector<Eigen::VectorXd>     & mu,            // size=NS, elm_size=dT
                           const std::vector<Eigen::MatrixXd>     & inv_Sigma,     // size=NS, elm_shape=(dT,dT)
                           double                                   tau,
                           const std::vector<Eigen::VectorXd>     & eta_batches,   // size=num_batches, elm_size=NT
                           const std::vector<int>                 & dirac_inds,    // size=num_diracs
                           const std::vector<double>              & dirac_weights, // size=num_diracs
                           const std::vector<int>                 & dirac2batch,   // size=num_diracs
                           const KDT::KDTree                      & dirac_kdtree,
                           unsigned long int                        num_neighbors)
{
    int dT             = target_coords[0].size();

    Eigen::VectorXd x = source_coords[source_ind];
    Eigen::VectorXd y = target_coords[target_ind];
    
    double          vol_x   = vol[source_ind];
    Eigen::VectorXd mu_x    = mu [source_ind];

    unsigned int num_neighbors2 = std::min(num_neighbors, dirac_kdtree.get_num_pts());

    Eigen::VectorXi nearest_diracs = dirac_kdtree.query( x, num_neighbors2 ).first;

    std::vector<int> nearest_diracs_list(nearest_diracs.size());
    for ( int ii=0; ii<nearest_diracs.size(); ++ii )
    {
        nearest_diracs_list[ii] = nearest_diracs(ii);
    }

    std::vector<std::pair<Eigen::VectorXd, double>> points_and_values;
    points_and_values.reserve(nearest_diracs.size());
    for ( int dd : nearest_diracs_list )
    {
        int jj = dirac_inds[dd];
        int b = dirac2batch[dd];

        double          weight_xj    = dirac_weights[dd];
        Eigen::VectorXd xj           = source_coords[jj];
        double          vol_xj       = vol[jj];
        Eigen::VectorXd mu_xj        = mu[jj];
        Eigen::MatrixXd inv_Sigma_xj = inv_Sigma[jj];

        Eigen::VectorXd dp = y - mu_x;
        Eigen::VectorXd z = dp + mu_xj;

        std::pair<Eigen::VectorXi, Eigen::MatrixXd> IC = target_mesh.first_point_collision( z );
        int             z_simplex_ind   = IC.first(0);
        Eigen::VectorXd z_affine_coords = IC.second.col(0);
        if ( z_simplex_ind >= 0 ) // point is good (z is in the mesh)
        {
            double kernel_value_estimate_from_xj = 0.0;
            if ( (dp.transpose() * (inv_Sigma_xj * dp )) < (tau * tau) )
            {
                double psi_j_at_z = 0.0;
                for ( int ii=0; ii<dT+1; ++ii )
                {
                    psi_j_at_z += z_affine_coords(ii) * eta_batches[b](target_mesh.cells(ii, z_simplex_ind));
                }
                psi_j_at_z /= weight_xj;

                kernel_value_estimate_from_xj = (vol_x / vol_xj) * psi_j_at_z;
            }
            points_and_values.push_back(std::make_pair(xj - x, kernel_value_estimate_from_xj));
        }
    }
    return points_and_values;
}

}