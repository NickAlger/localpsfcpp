#pragma once

#include <iostream>
#include <list>
#include <vector>

#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "kdtree.h"
#include "ellipsoid.h"


namespace IMPULSERESPONSE {

// apply_A    : L2(Omega_in)  -> L2'(Omega_out)
// apply_AT   : L2(Omega_out) -> L2'(Omega_in)
// apply_M_in : L2(Omega_in)  -> L2'(Omega_in)
// solve_M_in : L2'(Omega_in) -> L2(Omega_in)
std::tuple<std::vector<double>,          // all vols, V
           std::vector<Eigen::VectorXd>, // all means, mu
           std::vector<Eigen::MatrixXd>> // all covariances, Sigma
    compute_impulse_response_moments(const std::function<Eigen::VectorXd(Eigen::VectorXd)> & apply_AT,
                                     const std::function<Eigen::VectorXd(Eigen::VectorXd)> & solve_M_in,
                                     const Eigen::MatrixXd &dof_coords_out) // shape=(N_out, gdim_out)
{
    const int N_out = dof_coords_out.rows();
    const int d_out = dof_coords_out.cols();
    Eigen::VectorXd V = solve_M_in(apply_AT(Eigen::VectorXd::Ones(N_out)));
    const int N_in = V.size();

    Eigen::MatrixXd mu(N_in, d_out);
    for ( int ii=0; ii<d_out; ++ii )
    {
        mu.col(ii) = solve_M_in(apply_AT(dof_coords_out.col(ii))).array() / V.array();
    }

    Eigen::MatrixXd Sigma(N_in, d_out*d_out);
    for ( int ii=0; ii<d_out; ++ii )
    {
        for ( int jj=0; jj<=ii; ++jj)
        {
            const int ind1 = ii + d_out * jj;
            const int ind2 = jj + d_out * ii;
            Sigma.col(ind1) = solve_M_in(apply_AT(dof_coords_out.col(ii).array() * 
                                                  dof_coords_out.col(jj).array())).array() / V.array() 
                                - mu.col(ii).array() * mu.col(jj).array();
            Sigma.col(ind2) = Sigma.col(ind1);
        }
    }

    std::vector<double> all_vol(N_in);
    for ( int ii=0; ii<N_in; ++ii )
    {
        all_vol[ii] = V(ii);
    }

    std::vector<Eigen::VectorXd> all_mu(N_in);
    for ( int ii=0; ii<N_in; ++ii )
    {
        Eigen::VectorXd mu_i(d_out);
        for ( int jj=0; jj<d_out; ++jj )
        {
            mu_i(jj) = mu(ii,jj);
        }
        all_mu[ii] = mu_i;
    }

    std::vector<Eigen::MatrixXd> all_Sigma(N_in);
    for ( int ii=0; ii<N_in; ++ii )
    {
        Eigen::MatrixXd Sigma_i(d_out,d_out);
        for ( int jj=0; jj<d_out; ++jj )
        {
            for ( int kk=0; kk<d_out; ++kk )
            {
                Sigma_i(jj,kk) = Sigma(ii, jj + d_out * kk);
            }
        }
        all_Sigma[ii] = Sigma_i;
    }

    return std::make_tuple(all_vol, all_mu, all_Sigma);
}

// apply_A    : L2(Omega_in)   -> L2'(Omega_out)
// apply_AT   : L2(Omega_out)  -> L2'(Omega_in)
// apply_M_in : L2(Omega_in)   -> L2'(Omega_in)
// solve_M_in : L2'(Omega_in)  -> L2(Omega_in)
// apply_M_in : L2(Omega_out)  -> L2'(Omega_out)
// solve_M_in : L2'(Omega_out) -> L2(Omega_out)
Eigen::VectorXd compute_impulse_response_batch(const std::function<Eigen::VectorXd(Eigen::VectorXd)> & apply_A,
                                               const std::function<Eigen::VectorXd(Eigen::VectorXd)> & solve_M_in,
                                               const std::function<Eigen::VectorXd(Eigen::VectorXd)> & solve_M_out,
                                               const std::vector<int>    & dirac_inds,
                                               const std::vector<double> & dirac_weights,
                                               const double              & N_in)
{
    if (dirac_inds.size() != dirac_weights.size())
    {
        throw std::invalid_argument( "Different number of dirac_inds and dirac_weights" );
    }

    Eigen::VectorXd dirac_comb = Eigen::VectorXd::Zero(N_in);
    for ( int ii=0; ii<dirac_inds.size(); ++ii )
    {
        dirac_comb(dirac_inds[ii]) = dirac_weights[ii];
    }

    return solve_M_out(apply_A(solve_M_in(dirac_comb)));
}

// struct ImpulseResponseBatch
// {
//     Eigen::VectorXd            eta;
//     Eigen::MatrixXd            points;
//     Eigen::VectorXd            weights;
//     ELLIPSOID::EllipsoidForest ellipsoids;

//     int dS; // Spatial dimension of source space
//     int dT; // Spatial dimension of target space
//     int N;  // number of sample points

//     ImpulseResponseBatch( const Eigen::VectorXd            & eta_input, 
//                           const std::vector<int>           & inds,
//                           const std::vector<double>        & weights_input,
//                           const Eigen::MatrixXd            & source_coords,
//                           const ELLIPSOID::EllipsoidForest & target_EF )
//         : ellipsoids( target_EF, inds )
//     {
//         N = eta_input.size();
//         if ( inds.size() != N )
//         {
//             throw std::invalid_argument( "inds.size() != N" );
//         }
//         if ( weights_input.size() != N )
//         {
//             throw std::invalid_argument( "weights_input.size() != N" );
//         }

//         dS = source_coords.rows();
//         dT = target_EF.d;

//         eta = eta_input;
        
//         points.resize(dS, N);
//         for ( int ii=0; ii<N; ++ii )
//         {
//             points.col(ii) = source_coords.col(inds[ii]);
//         }

//         weights.resize(N);
//         for ( int ii=0; ii<N; ++ii )
//         {
//             weights(ii) = weights_input[ii];
//         }
//     }
// };

// struct ImpulseResponseBatches
// {
//     std::vector<ImpulseResponseBatch> batches;
//     std::vector<int>                  point2batch;
//     std::vector<int>                  batch2point_start;
//     std::vector<int>                  batch2point_stop;
//     KDT::KDTree                       kdtree;

//     int dS=-1; // Spatial dimension of source space
//     int dT=-1; // Spatial dimension of target space
//     int num_pts=0;
//     int num_batches=0;

//     void build_kdtree()
//     {
//         if ( num_pts > 0 )
//         {
//             Eigen::MatrixXd all_sample_points(dS, num_pts);
//             for ( int ii=0; ii<num_pts; ++ii )
//             {
//                 int b = point2batch[ii];
//                 int k = ii - batch2point_start[b];
//                 all_sample_points.col(k) = batches[b].points.col(k);
//             }
//             kdtree.build_tree(all_sample_points);
//         }
//     }

//     void add_batch(ImpulseResponseBatch & B)
//     {
//         if ( dS < 0 )
//         {
//             dS = B.dS;
//         }
//         else if ( B.dS != dS )
//         {
//             throw std::invalid_argument( "B.dS != dS" );
//         }
//         if ( dT < 0 )
//         {
//             dT = B.dT;
//         }
//         else if ( B.dT != dT )
//         {
//             throw std::invalid_argument( "B.dT != dT" );
//         }

//         for ( int ii=0; ii<B.N; ++ii )
//         {
//             point2batch.push_back(num_batches);
//         }
//         batch2point_start.push_back(num_pts);
//         batch2point_stop.push_back(num_pts + B.N);
//         batches.push_back(B);
//         num_pts += B.N;
//         num_batches += 1;

//         build_kdtree();
//     }
// };

}