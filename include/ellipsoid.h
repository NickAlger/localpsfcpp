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

inline bool point_is_in_ellipsoid( const Eigen::VectorXd mu,
                                   const Eigen::MatrixXd Sigma,
                                   const Eigen::VectorXd p,
                                   const double tau )
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

inline bool boxes_intersect( const Eigen::VectorXd A_min,
                             const Eigen::VectorXd A_max,
                             const Eigen::VectorXd B_min,
                             const Eigen::VectorXd B_max )
{
    return (( A_min.array() <= B_max.array() ).all() &&
            ( B_min.array() <= A_max.array() ).all());
}

double K_fct(const double s,
             const Eigen::VectorXd lambdas,
             const Eigen::VectorXd v,
             const double tau)
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

// dR = reference spatial dimension (e.g., 1, 2, or 3)
// dE = ellipsoid spatial dimension (e.g., 1, 2, or 3), 
// N = number of ellipsoids (e.g., thousands or millions)
struct EllipsoidForest
{
    Eigen::MatrixXd reference_points; // shape=(dR,    N)
    Eigen::VectorXd vol;              // shape=(N,)
    Eigen::MatrixXd mu;               // shape=(dE,    N)
    Eigen::MatrixXd Sigma;            // shape=(dE*dE, N)
    double          tau;

    int    dE;
    int    dR;
    int    N;

    Eigen::MatrixXd Sigma_eigenvectors; // shape=(dE*dE, N)
    Eigen::MatrixXd Sigma_eigenvalues;  // shape=(dE,    N)
    Eigen::MatrixXd iSigma;             // shape=(dE*dE, N)
    Eigen::MatrixXd sqrt_Sigma;         // shape=(dE*dE, N)
    Eigen::MatrixXd isqrt_Sigma;        // shape=(dE*dE, N)
    Eigen::VectorXd det_sqrt_Sigma;     // shape=(dE*dE, N)

    Eigen::MatrixXd box_mins;  // shape=(dE, N)
    Eigen::MatrixXd box_maxes; // shape=(dE, N)
    
    AABB::AABBTree ellipsoid_aabb;
    KDT::KDTree    reference_kdtree;

    void update_tau( double new_tau )
    {
        if ( new_tau < 0.0 )
        {
            throw std::invalid_argument( "negative tau given (tau must be non-negative)" );
        }
        tau = new_tau;

        for ( int ii=0; ii<N; ++ii )
        {
            Eigen::Map<const Eigen::MatrixXd> Sigma_i(Sigma.col(ii).data(), dE, dE);
            std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(mu.col(ii), Sigma_i, tau);
            box_mins.col(ii) = std::get<0>(B);
            box_maxes.col(ii) = std::get<1>(B);
        }
        ellipsoid_aabb = AABB::AABBTree(box_mins, box_maxes);
    }
};


EllipsoidForest create_ellipsoid_forest( const std::vector<Eigen::VectorXd> & reference_points_list,
                                         const std::vector<double>          & vol_list,
                                         const std::vector<Eigen::VectorXd> & mu_list,
                                         const std::vector<Eigen::MatrixXd> & Sigma_list,
                                         const double                         tau )
{
    if ( tau < 0.0 )
    {
        throw std::invalid_argument( "negative tau given (tau must be non-negative)" );
    }

    int N = mu_list.size();
    if (N <= 0)
    {
        throw std::invalid_argument( "No ellipsoids given. Cannot infer spatial dimension" );
    }
    if (reference_points_list.size() != N)
    {
        throw std::invalid_argument( "reference_points_list.size() != mu_list.size()" );
    }
    if (vol_list.size() != N)
    {
        throw std::invalid_argument( "vol_list.size() != mu_list.size()" );
    }
    if (Sigma_list.size() != N)
    {
        throw std::invalid_argument( "Sigma_list.size() != mu_list.size()" );
    }

    int dE = mu_list[0].size();
    int dR = reference_points_list[0].size();
    Eigen::MatrixXd reference_points(dR,    N);
    Eigen::VectorXd vol             (N);
    Eigen::MatrixXd mu              (dE,    N);
    Eigen::MatrixXd Sigma           (dE*dE, N);
    for ( int ii=0; ii<N; ++ii )
    {
        if ( reference_points_list[ii].size() != dR )
        {
            throw std::invalid_argument( "inconsistent sizes in reference_points_list" );
        }
        reference_points.col(ii) = reference_points_list[ii];

        vol(ii) = vol_list[ii];

        if ( mu_list[ii].size() != dE )
        {
            throw std::invalid_argument( "inconsistent sizes in mu_list" );
        }
        mu.col(ii) = mu_list[ii];

        if ( Sigma_list[ii].rows() != dE )
        {
            throw std::invalid_argument( "inconsistent row sizes in Sigma_list" );
        }
        if ( Sigma_list[ii].cols() != dE )
        {
            throw std::invalid_argument( "inconsistent col sizes in Sigma_list" );
        }
        Sigma.col(ii) = Eigen::Map<const Eigen::VectorXd>(Sigma_list[ii].data(), dE*dE);
    }

    Eigen::MatrixXd Sigma_eigenvectors(dE*dE, N);
    Eigen::MatrixXd Sigma_eigenvalues (dE,    N);
    Eigen::MatrixXd iSigma            (dE*dE, N);
    Eigen::MatrixXd sqrt_Sigma        (dE*dE, N);
    Eigen::MatrixXd isqrt_Sigma       (dE*dE, N);
    Eigen::VectorXd det_sqrt_Sigma    (N);
    for ( int ii=0; ii<N; ++ii )
    {
        Eigen::Map<const Eigen::MatrixXd> Sigma_i(Sigma.col(ii).data(), dE, dE);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma_i);
        Eigen::MatrixXd P   = es.eigenvectors();
        Eigen::MatrixXd iP  = P.inverse();

        Eigen::VectorXd dd       = es.eigenvalues();
        Eigen::VectorXd idd      = dd.array().inverse().matrix();
        Eigen::VectorXd sqrt_dd  = dd.array().sqrt().matrix();
        Eigen::VectorXd isqrt_dd = sqrt_dd.array().inverse().matrix();

        Eigen::MatrixXd iSigma_i         = P * idd.asDiagonal()      * iP;
        Eigen::MatrixXd sqrt_Sigma_i     = P * sqrt_dd.asDiagonal()  * iP;
        Eigen::MatrixXd isqrt_Sigma_i    = P * isqrt_dd.asDiagonal() * iP;

        Sigma_eigenvalues .col(ii) = dd;
        Sigma_eigenvectors.col(ii) = Eigen::Map<Eigen::VectorXd>(P            .data(), dE*dE);
        iSigma            .col(ii) = Eigen::Map<Eigen::VectorXd>(iSigma_i     .data(), dE*dE);
        sqrt_Sigma        .col(ii) = Eigen::Map<Eigen::VectorXd>(sqrt_Sigma_i .data(), dE*dE);
        isqrt_Sigma       .col(ii) = Eigen::Map<Eigen::VectorXd>(isqrt_Sigma_i.data(), dE*dE);

        det_sqrt_Sigma(ii) = sqrt_dd.prod();
    }

    Eigen::MatrixXd box_mins(dE, N);
    Eigen::MatrixXd box_maxes(dE, N);
    for ( int ii=0; ii<N; ++ii )
    {
        Eigen::Map<const Eigen::MatrixXd> Sigma_i(Sigma.col(ii).data(), dE, dE);
        std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(mu.col(ii), Sigma_i, tau);
        box_mins.col(ii) = std::get<0>(B);
        box_maxes.col(ii) = std::get<1>(B);
    }
    AABB::AABBTree ellipsoid_aabb(box_mins, box_maxes);

    KDT::KDTree reference_kdtree(reference_points);

    return EllipsoidForest{ reference_points, 
                            vol, mu, Sigma, tau, 
                            dE, dR, N,
                            Sigma_eigenvectors, Sigma_eigenvalues,
                            iSigma, sqrt_Sigma, isqrt_Sigma, det_sqrt_Sigma, 
                            box_mins, box_maxes,
                            ellipsoid_aabb, reference_kdtree };
}

std::tuple<std::vector<int>, std::vector<double>> // (new_batch, squared_distances)
    pick_ellipsoid_batch(const std::vector<std::vector<int>> & old_batches,
                         const std::vector<double>           & old_squared_distances,
                         const EllipsoidForest               & EF,
                         const double                        & min_vol_rtol)
{
    const double min_vol = min_vol_rtol * EF.vol.maxCoeff();

    std::vector<bool> is_pickable(EF.N);
    for ( int ii=0; ii<EF.N; ++ii )
    {
        is_pickable[ii] = (EF.vol[ii] > min_vol);
    }

    for ( std::vector<int> batch : old_batches)
    {
        for ( int k : batch)
        {
            is_pickable[k] = false;
        }
    }

    std::vector<int> candidate_inds(EF.N);
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
            Eigen::Map<const Eigen::MatrixXd> Sigma1(EF.Sigma.col(idx1).data(), EF.dE, EF.dE);
            std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(EF.mu.col(idx1), Sigma1, EF.tau);
            Eigen::VectorXi possible_collisions = EF.ellipsoid_aabb.box_collisions(std::get<0>(B), std::get<1>(B));
            for ( int jj=0; jj<possible_collisions.size(); ++jj )
            {
                int idx2 = possible_collisions[jj];
                if ( is_pickable[idx2] )
                {
                    Eigen::Map<const Eigen::MatrixXd> Sigma2(EF.Sigma.col(idx2).data(), EF.dE, EF.dE);
                    if ( ellipsoids_intersect(EF.mu.col(idx2), Sigma2,
                                              EF.mu.col(idx1), Sigma1,
                                              EF.tau) )
                    {
                        is_pickable[idx2] = false;
                    }
                }
            }
        }
    }

    std::vector<double> squared_distances = old_squared_distances;
    for ( int ind : next_batch )
    {
        for ( int ii=0; ii<EF.N; ++ii )
        {
            double old_dsq = old_squared_distances[ii];
            double new_dsq = (EF.reference_points.col(ind) - EF.reference_points.col(ii)).squaredNorm();
            if ( new_dsq < old_dsq || old_dsq < 0.0 )
            {
                squared_distances[ii] = new_dsq;
            }
        }
    }

    return std::make_tuple(next_batch, squared_distances);
}


} // end namespace ELLIPSOID