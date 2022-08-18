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
double RBF_TPS_interpolate( const Eigen::VectorXd & function_at_rbf_points,
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

// Radial basis function interpolation with Gaussian kernel basis functions
double RBF_GAUSS_interpolate( const Eigen::VectorXd & function_at_rbf_points,
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
        Eigen::VectorXd max_pt = rbf_points.rowwise().maxCoeff();
        Eigen::VectorXd min_pt  = rbf_points.rowwise().minCoeff();

        double diam_squared = (max_pt - min_pt).squaredNorm();
        double sigma_squared = diam_squared / (3.0 * 3.0);

        Eigen::MatrixXd M(N, N);
        for ( int jj=0; jj<N; ++jj )
        {
            for ( int ii=0; ii<N; ++ii )
            {
                double r_squared = (rbf_points.col(ii) - rbf_points.col(jj)).squaredNorm();
                M(ii,jj) = exp( - 0.5 * r_squared / sigma_squared);
            }
        }

        Eigen::VectorXd weights = M.lu().solve(function_at_rbf_points);

        Eigen::VectorXd rbfs_at_eval_point(N);
        for ( int ii=0; ii<N; ++ii )
        {
            double r_squared = (rbf_points.col(ii) - eval_point).squaredNorm();
            rbfs_at_eval_point(ii) = exp( - 0.5 * r_squared / sigma_squared);
        }

        function_at_eval_point = (weights.array() * rbfs_at_eval_point.array()).sum();
    }
    return function_at_eval_point;
}

inline double eval_simplexmesh( const Eigen::VectorXd & f,
                                const Eigen::VectorXd & z_affine_coords, 
                                unsigned long int       z_simplex_ind,
                                const Eigen::MatrixXi & mesh_cells )
{
    double f_of_z = 0.0;
    for ( int ii=0; ii<z_affine_coords.size(); ++ii )
    {
        f_of_z += z_affine_coords(ii) * f(mesh_cells(ii, z_simplex_ind));
    }
    return f_of_z;
}

inline std::vector<int> find_nearest_diracs( const Eigen::VectorXd & x,
                                             const KDT::KDTree     & dirac_kdtree, 
                                             unsigned long int       num_neighbors )
{
    unsigned int num_neighbors2 = std::min(num_neighbors, dirac_kdtree.get_num_pts());

    Eigen::VectorXi nearest_diracs_VectorXi = dirac_kdtree.query( x, num_neighbors2 ).first;

    std::vector<int> nearest_diracs(nearest_diracs_VectorXi.size());
    for ( unsigned int ii=0; ii<nearest_diracs.size(); ++ii )
    {
        nearest_diracs[ii] = nearest_diracs_VectorXi(ii);
    }
    return nearest_diracs;
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
    Eigen::VectorXd x = source_coords[source_ind];
    Eigen::VectorXd y = target_coords[target_ind];
    
    double          vol_x   = vol[source_ind];
    Eigen::VectorXd mu_x    = mu [source_ind];

    std::vector<int> nearest_diracs = find_nearest_diracs(x, dirac_kdtree, num_neighbors );

    std::vector<std::pair<Eigen::VectorXd, double>> points_and_values;
    points_and_values.reserve(nearest_diracs.size());
    for ( int dd : nearest_diracs )
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
                double psi_j_at_z = eval_simplexmesh(eta_batches[b], z_affine_coords, z_simplex_ind, target_mesh.cells) / weight_xj;
                kernel_value_estimate_from_xj = (vol_x / vol_xj) * psi_j_at_z;
            }
            points_and_values.push_back(std::make_pair(xj - x, kernel_value_estimate_from_xj));
        }
    }
    return points_and_values;
}

enum class ShiftMethod { LOW_RANK,                           // z = y, 
                         LOCAL_TRANSLATION_INVARIANCE,       // z = x_j  + y - x
                         LOCAL_MEAN_DISPLACEMENT_INVARIANCE, // z = mu_j + y - mu_x
                         ELLIPSOID_MAPPING                   // z = mu_j + sqrt_Sigma_xj * inv_sqrt_Sigma_x * (y - mu_x)
                       };

enum class ScalingMethod { NONE,  // w = 1.0
                           VOL,   // w = vol_x / vol_xj
                           DET,   // w = det_sqrt_Sigma_xj / det_sqrt_Sigma_x
                           DETVOL // w = (vol_x / vol_xj) * (det_sqrt_Sigma_xj / det_sqrt_Sigma_x)
                         };

enum class InterpolationMethod { RBF_THIN_PLATE_SPLINES, // Thin plate spline radial basis function interpolation
                                 RBF_GAUSS };            // Gaussian radial basis function interpolation

// Local local translation invariance points and values
std::vector<std::pair<Eigen::VectorXd, double>> 
    interpolation_points_and_values(unsigned long int                        target_ind,
                                    unsigned long int                        source_ind,
                                    const std::vector<Eigen::VectorXd>     & source_coords,  // size=NS, elm_size=dS
                                    const std::vector<Eigen::VectorXd>     & target_coords,  // size=NT, elm_size=dT
                                    const SMESH::SimplexMesh               & target_mesh,
                                    const std::vector<double>              & vol,            // size=NS
                                    const std::vector<Eigen::VectorXd>     & mu,             // size=NS, elm_size=dT
                                    const std::vector<Eigen::MatrixXd>     & inv_Sigma,      // size=NS, elm_shape=(dT,dT)
                                    const std::vector<Eigen::MatrixXd>     & sqrt_Sigma,     // size=NS, elm_shape=(dT,dT)
                                    const std::vector<Eigen::MatrixXd>     & inv_sqrt_Sigma, // size=NS, elm_shape=(dT,dT)
                                    const std::vector<double>              & det_sqrt_Sigma, // size=NS
                                    double                                   tau,
                                    const std::vector<Eigen::VectorXd>     & eta_batches,    // size=num_batches, elm_size=NT
                                    const std::vector<int>                 & dirac_inds,     // size=num_diracs
                                    const std::vector<double>              & dirac_weights,  // size=num_diracs
                                    const std::vector<int>                 & dirac2batch,    // size=num_diracs
                                    const KDT::KDTree                      & dirac_kdtree,
                                    unsigned long int                        num_neighbors,
                                    ShiftMethod                              shift_method,
                                    ScalingMethod                            scaling_method)
{
    unsigned int dT = target_coords[0].size();

    Eigen::VectorXd x = source_coords[source_ind];
    Eigen::VectorXd y = target_coords[target_ind];
    
    double          vol_x            = vol           [source_ind];
    Eigen::VectorXd mu_x             = mu            [source_ind];
    Eigen::MatrixXd inv_sqrt_Sigma_x = inv_sqrt_Sigma[source_ind];
    double          det_sqrt_Sigma_x = det_sqrt_Sigma[source_ind];

    if ( (scaling_method == ScalingMethod::DET || scaling_method == ScalingMethod::DETVOL) && det_sqrt_Sigma_x == 0.0 )
    {
        std::vector<std::pair<Eigen::VectorXd, double>> points_and_values;
        points_and_values.push_back(std::make_pair(x - x, 0.0));
        return points_and_values;
    }

    std::vector<int> nearest_diracs = find_nearest_diracs(x, dirac_kdtree, num_neighbors );

    std::vector<std::pair<Eigen::VectorXd, double>> points_and_values;
    points_and_values.reserve(nearest_diracs.size());
    for ( int dd : nearest_diracs )
    {
        int jj = dirac_inds[dd];
        int b = dirac2batch[dd];

        double dirac_weight_xj = dirac_weights[dd];

        Eigen::VectorXd xj                = source_coords [jj];
        double          vol_xj            = vol           [jj];
        Eigen::VectorXd mu_xj             = mu            [jj];
        Eigen::MatrixXd sqrt_Sigma_xj     = sqrt_Sigma    [jj];
        Eigen::MatrixXd inv_Sigma_xj      = inv_Sigma     [jj];
        double          det_sqrt_Sigma_xj = det_sqrt_Sigma[jj];

        Eigen::VectorXd z(dT);
        switch( shift_method ) 
        {
            case ShiftMethod::LOW_RANK:                           z = y;                                                       break;
            case ShiftMethod::LOCAL_TRANSLATION_INVARIANCE:       z = xj + y - x;                                              break;
            case ShiftMethod::LOCAL_MEAN_DISPLACEMENT_INVARIANCE: z = mu_xj + y - mu_x;                                        break;
            case ShiftMethod::ELLIPSOID_MAPPING:                  z = mu_xj + sqrt_Sigma_xj * (inv_sqrt_Sigma_x * (y - mu_x)); break;
        }

        std::pair<Eigen::VectorXi, Eigen::MatrixXd> IC = target_mesh.first_point_collision( z );
        int             z_simplex_ind   = IC.first(0);
        Eigen::VectorXd z_affine_coords = IC.second.col(0);
        if ( z_simplex_ind >= 0 ) // point is good (z is in the mesh)
        {
            double kernel_value_estimate_from_xj = 0.0;
            Eigen::VectorXd dp = z - mu_xj;
            bool z_is_in_Ej = ( (dp.transpose() * (inv_Sigma_xj * dp )) < (tau * tau) );
            if ( z_is_in_Ej )
            {
                double psi_j_at_z = eval_simplexmesh(eta_batches[b], z_affine_coords, z_simplex_ind, target_mesh.cells) / dirac_weight_xj;

                double w = 0.0;
                switch( scaling_method ) 
                {
                    case ScalingMethod::NONE:   w = 1.0;                                                           break;
                    case ScalingMethod::VOL:    w = vol_x / vol_xj;                                                break;
                    case ScalingMethod::DET:    w = det_sqrt_Sigma_xj / det_sqrt_Sigma_x;                          break;
                    case ScalingMethod::DETVOL: w = ( vol_x / vol_xj ) * ( det_sqrt_Sigma_xj / det_sqrt_Sigma_x ); break;
                }

                kernel_value_estimate_from_xj = w * psi_j_at_z;
            }
            points_and_values.push_back(std::make_pair(xj - x, kernel_value_estimate_from_xj));
        }
    }
    return points_and_values;
}


}