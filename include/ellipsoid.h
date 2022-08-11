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


struct EllipsoidForest
{
    Eigen::VectorXd vol;              // shape=(N,)
    Eigen::MatrixXd mu;               // shape=(d,    N)
    Eigen::MatrixXd Sigma;            // shape=(d*d, N)
    double          tau;

    int d; // reference spatial dimension (e.g., 1, 2, or 3)
    int N;  // number of ellipsoids (e.g., thousands or millions)

    Eigen::MatrixXd Sigma_eigenvectors; // shape=(d*d, N)
    Eigen::MatrixXd Sigma_eigenvalues;  // shape=(d,    N)
    Eigen::MatrixXd iSigma;             // shape=(d*d, N)
    Eigen::MatrixXd sqrt_Sigma;         // shape=(d*d, N)
    Eigen::MatrixXd isqrt_Sigma;        // shape=(d*d, N)
    Eigen::VectorXd det_sqrt_Sigma;     // shape=(N,)

    Eigen::MatrixXd box_mins;  // shape=(d, N)
    Eigen::MatrixXd box_maxes; // shape=(d, N)
    
    AABB::AABBTree ellipsoid_aabb;

    EllipsoidForest( const std::vector<double>          & vol_list,
                     const std::vector<Eigen::VectorXd> & mu_list,
                     const std::vector<Eigen::MatrixXd> & Sigma_list,
                     const double                         initial_tau )
    {
        if ( initial_tau < 0.0 )
        {
            throw std::invalid_argument( "negative initial_tau given" );
        }
        tau = initial_tau;

        N = mu_list.size();
        if (N <= 0)
        {
            throw std::invalid_argument( "No ellipsoids given. Cannot infer spatial dimension" );
        }
        if (vol_list.size() != N)
        {
            throw std::invalid_argument( "vol_list.size() != mu_list.size()" );
        }
        if (Sigma_list.size() != N)
        {
            throw std::invalid_argument( "Sigma_list.size() != mu_list.size()" );
        }

        d = mu_list[0].size();
        vol             .resize(N);
        mu              .resize(d,    N);
        Sigma           .resize(d*d, N);
        for ( int ii=0; ii<N; ++ii )
        {
            vol(ii) = vol_list[ii];

            if ( mu_list[ii].size() != d )
            {
                throw std::invalid_argument( "inconsistent sizes in mu_list" );
            }
            mu.col(ii) = mu_list[ii];

            if ( Sigma_list[ii].rows() != d )
            {
                throw std::invalid_argument( "inconsistent row sizes in Sigma_list" );
            }
            if ( Sigma_list[ii].cols() != d )
            {
                throw std::invalid_argument( "inconsistent col sizes in Sigma_list" );
            }
            Sigma.col(ii) = Eigen::Map<const Eigen::VectorXd>(Sigma_list[ii].data(), d*d);
        }

        Sigma_eigenvectors.resize(d*d, N);
        Sigma_eigenvalues .resize(d,   N);
        iSigma            .resize(d*d, N);
        sqrt_Sigma        .resize(d*d, N);
        isqrt_Sigma       .resize(d*d, N);
        det_sqrt_Sigma    .resize(N);
        for ( int ii=0; ii<N; ++ii )
        {
            Eigen::Map<const Eigen::MatrixXd> Sigma_i(Sigma.col(ii).data(), d, d);
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
            Sigma_eigenvectors.col(ii) = Eigen::Map<Eigen::VectorXd>(P            .data(), d*d);
            iSigma            .col(ii) = Eigen::Map<Eigen::VectorXd>(iSigma_i     .data(), d*d);
            sqrt_Sigma        .col(ii) = Eigen::Map<Eigen::VectorXd>(sqrt_Sigma_i .data(), d*d);
            isqrt_Sigma       .col(ii) = Eigen::Map<Eigen::VectorXd>(isqrt_Sigma_i.data(), d*d);

            det_sqrt_Sigma(ii) = sqrt_dd.prod();
        }

        update_bounding_boxes();
        ellipsoid_aabb.build_tree(box_mins, box_maxes);
    }

    // Create a subforest from a bigger forest
    EllipsoidForest( const EllipsoidForest & bigger_EF, const std::vector<int> & subforest_inds )
    {
        d = bigger_EF.d;

        N = subforest_inds.size();
        for ( int ii=0; ii<N; ++ii )
        {
            if ( subforest_inds[ii] < 0 )
            {
                throw std::invalid_argument( "subforest_inds[ii] < 0" );
            }
            if ( subforest_inds[ii] >= bigger_EF.N )
            {
                throw std::invalid_argument( "subforest_inds[ii] >= bigger_EF.N" );
            }
        }

        tau = bigger_EF.tau;
        
        vol  .resize(N);
        det_sqrt_Sigma.resize(N);

        mu   .resize(d,   N);
        Sigma.resize(d*d, N);
        Sigma_eigenvectors.resize(d*d, N);
        Sigma_eigenvalues.resize(d, N);
        iSigma.resize(d*d, N);
        sqrt_Sigma.resize(d*d, N);
        isqrt_Sigma.resize(d*d, N);
        box_mins.resize(d, N);
        box_maxes.resize(d, N);
        for ( int ii=0; ii<N; ++ii )
        {
            int ind = subforest_inds[ii]

            vol           (ii) = bigger_EF.vol           (ind);
            det_sqrt_Sigma(ii) = bigger_EF.det_sqrt_Sigma(ind);

            mu                .col(ii) = bigger_EF   .mu                .col(ind);
            Sigma             .col(ii) = bigger_EF   .Sigma             .col(ind);
            Sigma_eigenvectors.col(ii) = bigger_EF   .Sigma_eigenvectors.col(ind);
            Sigma_eigenvalues .col(ii) = bigger_EF   .Sigma_eigenvalues .col(ind);
            iSigma            .col(ii) = bigger_EF   .iSigma            .col(ind);
            sqrt_Sigma        .col(ii) = bigger_EF   .sqrt_Sigma        .col(ind);
            isqrt_Sigma       .col(ii) = bigger_EF   .isqrt_Sigma       .col(ind);
            box_mins          .col(ii) = bigger_EF   .box_mins          .col(ind);
            box_maxes         .col(ii) = bigger_EF   .box_maxes         .col(ind);
        }

        ellipsoid_aabb.build_tree(box_mins, box_maxes);
    } 

    void update_bounding_boxes()
    {
        box_mins .resize(d, N);
        box_maxes.resize(d, N);
        for ( int ii=0; ii<N; ++ii )
        {
            Eigen::MatrixXd Sigma_i = Eigen::Map<Eigen::MatrixXd>(Sigma.col(ii).data(), d, d);
            std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(mu.col(ii), Sigma_i, tau);
            box_mins .col(ii) = std::get<0>(B);
            box_maxes.col(ii) = std::get<1>(B);
        }
    }

    void update_tau( double new_tau )
    {
        if ( new_tau < 0.0 )
        {
            throw std::invalid_argument( "negative tau given" );
        }
        tau = new_tau;

        update_bounding_boxes();
        ellipsoid_aabb.build_tree(box_mins, box_maxes);
    }

    std::tuple<std::vector<int>, std::vector<double>> // (new_batch, squared_distances)
        pick_ellipsoid_batch(const std::vector<std::vector<int>> & old_batches,
                             const std::vector<double>           & old_squared_distances, // size=N
                             const std::vector<Eigen::VectorXd>  & reference_points, // size=N
                             const double                        & min_vol_rtol)
    {
        const double min_vol = min_vol_rtol * vol.maxCoeff();

        std::vector<bool> is_pickable(N);
        for ( int ii=0; ii<N; ++ii )
        {
            is_pickable[ii] = (vol[ii] > min_vol);
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
                Eigen::Map<const Eigen::MatrixXd> Sigma1(Sigma.col(idx1).data(), d, d);
                std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(mu.col(idx1), Sigma1, tau);
                Eigen::VectorXi possible_collisions = ellipsoid_aabb.box_collisions(std::get<0>(B), std::get<1>(B));
                for ( int jj=0; jj<possible_collisions.size(); ++jj )
                {
                    int idx2 = possible_collisions[jj];
                    if ( is_pickable[idx2] )
                    {
                        Eigen::Map<const Eigen::MatrixXd> Sigma2(Sigma.col(idx2).data(), d, d);
                        if ( ellipsoids_intersect(mu.col(idx2), Sigma2,
                                                  mu.col(idx1), Sigma1,
                                                  tau) )
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
            for ( int ii=0; ii<N; ++ii )
            {
                double old_dsq = old_squared_distances[ii];
                double new_dsq = (reference_points[ind] - reference_points[ii]).squaredNorm();
                if ( new_dsq < old_dsq || old_dsq < 0.0 )
                {
                    squared_distances[ii] = new_dsq;
                }
            }
        }

        return std::make_tuple(next_batch, squared_distances);
    }

};


} // end namespace ELLIPSOID