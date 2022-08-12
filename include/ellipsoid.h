#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <numeric>
#include <algorithm>

#include <math.h>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "brent_minimize.h"
#include "aabbtree.h"
#include "kdtree.h"

namespace ELLIPSOID {

inline bool point_is_in_ellipsoid( const Eigen::VectorXd & mu,
                                   const Eigen::MatrixXd & Sigma,
                                   const Eigen::VectorXd & p,
                                   const double          & tau )
{
    Eigen::VectorXd z = p - mu;
    return Sigma.ldlt().solve(z).dot(z) <= (tau * tau);
}

inline std::tuple<Eigen::VectorXd, Eigen::VectorXd> ellipsoid_bounding_box( const Eigen::VectorXd & mu,
                                                                            const Eigen::MatrixXd & Sigma,
                                                                            const double          & tau )
{
    Eigen::VectorXd diag_Sigma = Sigma.diagonal();
    Eigen::VectorXd widths = (diag_Sigma.array().sqrt() * tau).matrix();
    return std::make_tuple(mu - widths, mu + widths);
}

inline bool boxes_intersect( const Eigen::VectorXd & A_min,
                             const Eigen::VectorXd & A_max,
                             const Eigen::VectorXd & B_min,
                             const Eigen::VectorXd & B_max )
{
    return (( A_min.array() <= B_max.array() ).all() &&
            ( B_min.array() <= A_max.array() ).all());
}

double K_fct(const double & s,
             const Eigen::VectorXd & lambdas,
             const Eigen::VectorXd & v,
             const double          & tau)
{
    double K = 0.0;
    for ( int ii=0; ii<lambdas.size(); ++ii )
    {
        K += (v(ii) * v(ii)) / ( 1.0 + s * (lambdas(ii) - 1.0) );
    }
    K = 1.0 - ( s * (1.0 - s) / (tau * tau) ) * K;
    return K;
}

bool ellipsoids_intersect( const Eigen::VectorXd & mu_A,
                           const Eigen::MatrixXd & Sigma_A,
                           const Eigen::VectorXd & mu_B,
                           const Eigen::MatrixXd & Sigma_B,
                           const double          & tau )
{
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> A_box = ellipsoid_bounding_box(mu_A, Sigma_A, tau);
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> B_box = ellipsoid_bounding_box(mu_B, Sigma_B, tau);
//    if (true)
    if (boxes_intersect(std::get<0>(A_box), std::get<1>(A_box),
                        std::get<0>(B_box), std::get<1>(B_box)))
    {
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma_A, Sigma_B);
        Eigen::VectorXd lambdas = es.eigenvalues();
        Eigen::VectorXd v = es.eigenvectors().transpose() * (mu_A - mu_B);

        std::function<double(double)> K = [lambdas, v, tau](const double s) {return K_fct(s, lambdas, v, tau);};

        std::tuple<double, double, int, int> sol = BRENT::brent_minimize( K, 0.0, 1.0, 1.0e-8, 200 );
        return (std::get<1>(sol) >= 0);
    }
    else
    {
        return false;
    }
}


std::tuple< Eigen::MatrixXd,  // Sigma
            bool,             // Sigma_is_good
            Eigen::MatrixXd,  // inv_Sigma
            Eigen::MatrixXd,  // sqrt_Sigma;
            Eigen::MatrixXd,  // inv_sqrt_Sigma
            double,           // det_sqrt_Sigma
            Eigen::VectorXd,  // unmodified_eigvals
            Eigen::MatrixXd > // unmodified_eigvecs
    postprocess_covariance( const Eigen::MatrixXd & Sigma0 )
{
    int d = Sigma0.rows();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma0);
    Eigen::VectorXd ee = es.eigenvalues();
    Eigen::MatrixXd P  = es.eigenvectors();

    Eigen::VectorXd unmodified_eigvals = ee;
    Eigen::MatrixXd unmodified_eigvecs = P;

    bool Sigma0_is_good = true;
    Eigen::VectorXd iee(d);
    Eigen::VectorXd sqrt_ee(d);
    Eigen::VectorXd isqrt_ee(d);
    for ( int kk=0; kk<d; ++kk )
    {
        if ( ee(kk) < 0.0 )
        {
            Sigma0_is_good = false;
            ee(kk)  = 0.0;
            iee(kk) = std::numeric_limits<double>::infinity();
            sqrt_ee(kk) = 0.0;
            isqrt_ee(kk) = std::numeric_limits<double>::infinity();
        }
        else
        {
            iee(kk) = 1.0 / ee(kk);
            sqrt_ee(kk) = sqrt(ee(kk));
            isqrt_ee(kk) = 1.0 / sqrt_ee(kk);
        }
    }

    Eigen::MatrixXd iP  = P.inverse(); // transpose should work too?

    Eigen::MatrixXd Sigma          = P * ee.asDiagonal()       * iP;
    Eigen::MatrixXd inv_Sigma      = P * iee.asDiagonal()      * iP;
    Eigen::MatrixXd sqrt_Sigma     = P * sqrt_ee.asDiagonal()  * iP;
    Eigen::MatrixXd inv_sqrt_Sigma = P * isqrt_ee.asDiagonal() * iP;

    double det_sqrt_Sigma = sqrt_ee.prod();

    return std::make_tuple(Sigma, Sigma0_is_good, 
                           inv_Sigma, sqrt_Sigma, inv_sqrt_Sigma, det_sqrt_Sigma, 
                           unmodified_eigvals, unmodified_eigvecs);
}


AABB::AABBTree make_ellipsoid_aabbtree(const std::vector<Eigen::VectorXd>  & mu, // size=N, elm_size=d
                                       const std::vector<Eigen::MatrixXd>  & Sigma, // size=N, elm_shape=(d,d)
                                       double                                tau)
{
    if ( tau < 0.0 )
    {
        throw std::invalid_argument( "negative initial_tau given" );
    }

    int N = mu.size();
    if (N <= 0)
    {
        throw std::invalid_argument( "No ellipsoids given. Cannot infer spatial dimension" );
    }
    int d = mu[0].size();

    if (Sigma.size() != mu.size())
    {
        throw std::invalid_argument( "Sigma.size() != mu.size()" );
    }
    
    Eigen::MatrixXd box_mins (d, N);
    Eigen::MatrixXd box_maxes(d, N);
    for ( int ii=0; ii<N; ++ii )
    {
        std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(mu[ii], Sigma[ii], tau);
        box_mins .col(ii) = std::get<0>(B);
        box_maxes.col(ii) = std::get<1>(B);
    }
    return AABB::AABBTree(box_mins, box_maxes);
}


std::tuple<std::vector<int>, std::vector<double>> // (new_batch, squared_distances)
    pick_ellipsoid_batch(const std::vector<std::vector<int>> & old_batches,
                         const std::vector<double>           & old_squared_distances, // size=N
                         const std::vector<Eigen::VectorXd>  & reference_points, // size=N
                         const std::vector<double>           & vol, // size=N
                         const std::vector<Eigen::VectorXd>  & mu, // size=N, elm_size=d
                         const std::vector<Eigen::MatrixXd>  & Sigma, // size=N, elm_shape=(d,d)
                         const std::vector<bool>             & Sigma_is_good, // size=N
                         double                                tau,
                         const AABB::AABBTree                & ellipsoid_aabb,
                         double                                min_vol_rtol)
{
    int N = vol.size();

    double max_vol = 0.0;
    for ( int ii=0; ii<N; ++ii )
    {
        max_vol = std::max(vol[ii], max_vol);
    }
    const double min_vol = min_vol_rtol * max_vol;

    std::vector<bool> is_pickable(N);
    for ( int ii=0; ii<N; ++ii )
    {
        is_pickable[ii] = ( (vol[ii] > min_vol) && Sigma_is_good[ii] );
    }

    for ( std::vector<int> batch : old_batches)
    {
        for ( int k : batch)
        {
            is_pickable[k] = false;
        }
    }

    std::vector<int> candidate_inds(N);
    std::iota(candidate_inds.begin(), candidate_inds.end(), 0);
    stable_sort(candidate_inds.begin(), candidate_inds.end(),
        [&old_squared_distances](int i1, int i2) 
            {return old_squared_distances[i1] > old_squared_distances[i2];});

    std::vector<int> next_batch;
    for ( int idx1 : candidate_inds )
    {
        if ( is_pickable[idx1] )
        {
            next_batch.push_back(idx1);
            is_pickable[idx1] = false;
            std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(mu[idx1], Sigma[idx1], tau);
            Eigen::VectorXi possible_collisions = ellipsoid_aabb.box_collisions(std::get<0>(B), std::get<1>(B));
            for ( int jj=0; jj<possible_collisions.size(); ++jj )
            {
                int idx2 = possible_collisions[jj];
                if ( is_pickable[idx2] )
                {
                    if ( ellipsoids_intersect(mu[idx2], Sigma[idx2],
                                              mu[idx1], Sigma[idx1],
                                              tau) )
                    {
                        is_pickable[idx2] = false;
                    }
                }
            }
        }
    }

    std::vector<double> squared_distances(N);
    for ( int ii=0; ii<N; ++ii )
    {
        squared_distances[ii] = old_squared_distances[ii];
        for ( int ind : next_batch )
        {
            double old_dsq = squared_distances[ii];
            double new_dsq = (reference_points[ind] - reference_points[ii]).squaredNorm();
            if ( new_dsq < old_dsq || old_dsq < 0.0 )
            {
                squared_distances[ii] = new_dsq;
            }
        }
    }

    return std::make_tuple(next_batch, squared_distances);
}


} // end namespace ELLIPSOID