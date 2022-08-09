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

inline std::tuple<Eigen::VectorXd, Eigen::VectorXd> ellipsoid_bounding_box( const Eigen::VectorXd mu,
                                                                            const Eigen::MatrixXd Sigma,
                                                                            const double tau )
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

bool ellipsoids_intersect( const Eigen::VectorXd mu_A,
                           const Eigen::MatrixXd Sigma_A,
                           const Eigen::VectorXd mu_B,
                           const Eigen::MatrixXd Sigma_B,
                           double tau )
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


inline Eigen::MatrixXd sqrtm(const Eigen::MatrixXd & A)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    const Eigen::MatrixXd P  = es.eigenvectors();
    const Eigen::MatrixXd sqrt_D = es.eigenvalues().array().sqrt().matrix().asDiagonal();
    return P * sqrt_D  * P.inverse();
}

struct EllipsoidForest
{
    std::vector<Eigen::VectorXd> reference_points;
    std::vector<double>          vol;
    std::vector<Eigen::VectorXd> mu;
    std::vector<Eigen::MatrixXd> Sigma;
    double                       tau;

    int    ellipsoid_gdim;
    int    reference_gdim;
    int    num_pts;
    double max_vol;

    std::vector<Eigen::MatrixXd> sqrt_Sigma;
    std::vector<Eigen::MatrixXd> isqrt_Sigma;
    std::vector<double>          det_sqrt_Sigma;
    
    AABB::AABBTree               ellipsoid_aabb;
    KDT::KDTree                  reference_kdtree;

    EllipsoidForest( const std::vector<Eigen::VectorXd> & reference_points,
                     const std::vector<double>          & vol,
                     const std::vector<Eigen::VectorXd> & mu,
                     const std::vector<Eigen::MatrixXd> & Sigma,
                     const double                       & tau )
        : reference_points(reference_points), vol(vol), mu(mu), Sigma(Sigma), tau(tau)
    {
        num_pts = mu.size();
        ellipsoid_gdim = mu[0].size();
        reference_gdim = reference_points[0].size();

        max_vol = 0.0;
        for ( int ii=0; ii<num_pts; ++ii )
        {
            if ( vol[ii] > max_vol )
            {
                max_vol = vol[ii];
            }
        }

        sqrt_Sigma     .resize(num_pts);
        isqrt_Sigma    .resize(num_pts);
        det_sqrt_Sigma .resize(num_pts);
        for ( int ii=0; ii<num_pts; ++ii )
        {
            sqrt_Sigma[ii]     = sqrtm(Sigma[ii]);
            isqrt_Sigma[ii]    = sqrt_Sigma[ii].inverse(); // inefficient, but who cares because matrix is small
            det_sqrt_Sigma[ii] = sqrt_Sigma[ii].determinant();
        }
        
        Eigen::MatrixXd box_mins(ellipsoid_gdim, num_pts);
        Eigen::MatrixXd box_maxes(ellipsoid_gdim, num_pts);
        for ( int ii=0; ii<num_pts; ++ii )
        {
            std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(mu[ii], Sigma[ii], tau);
            box_mins.col(ii) = std::get<0>(B);
            box_maxes.col(ii) = std::get<1>(B);
        }
        ellipsoid_aabb.build_tree(box_mins, box_maxes);

        Eigen::MatrixXd reference_points_matrix(ellipsoid_gdim, num_pts);
        for ( int ii=0; ii<num_pts; ++ii )
        {
            reference_points_matrix.col(ii) = reference_points[ii];
        }
        reference_kdtree.build_tree(reference_points_matrix);
    }
};

std::tuple<std::vector<int>, std::vector<double>> // (new_batch, squared_distances)
    pick_ellipsoid_batch(const std::vector<std::vector<int>> & old_batches,
                         const std::vector<double>           & old_squared_distances,
                         const EllipsoidForest               & EF,
                         const double                        & min_vol_rtol)
{
    const double min_vol = min_vol_rtol * EF.max_vol;

    std::vector<bool> is_pickable(EF.num_pts);
    for ( int ii=0; ii<EF.num_pts; ++ii )
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

    std::vector<int> candidate_inds(EF.num_pts);
    std::iota(candidate_inds.begin(), candidate_inds.end(), 0);
    stable_sort(candidate_inds.begin(), candidate_inds.end(),
        [&old_squared_distances](int i1, int i2) 
            {return old_squared_distances[i1] > old_squared_distances[i2];});

    std::vector<int> next_batch;
    for ( int idx : candidate_inds )
    {
        if ( is_pickable[idx] )
        {
            next_batch.push_back(idx);
            is_pickable[idx] = false;
            std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(EF.mu[idx], EF.Sigma[idx], EF.tau);
            Eigen::VectorXi possible_collisions = EF.ellipsoid_aabb.box_collisions(std::get<0>(B), std::get<1>(B));
            for ( int jj=0; jj<possible_collisions.size(); ++jj )
            {
                int idx2 = possible_collisions[jj];
                if ( is_pickable[idx2] )
                {
                    if ( ellipsoids_intersect(EF.mu[idx2], EF.Sigma[idx2],
                                              EF.mu[idx],  EF.Sigma[idx],
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
        for ( int ii=0; ii<EF.num_pts; ++ii )
        {
            double old_dsq = old_squared_distances[ii];
            double new_dsq = (EF.reference_points[ind] - EF.reference_points[ii]).squaredNorm();
            if ( new_dsq < old_dsq || old_dsq < 0.0 )
            {
                squared_distances[ii] = new_dsq;
            }
        }
    }

    return std::make_tuple(next_batch, squared_distances);
}


class EllipsoidBatchPicker
{
private:
    AABB::AABBTree               aabb;
    std::vector<Eigen::VectorXd> all_points;
    std::vector<double>          all_vol;
    std::vector<Eigen::VectorXd> all_mu;
    std::vector<Eigen::MatrixXd> all_Sigma;
    int                          num_pts;
    int                          spatial_dim;
    double                       tau;
    double                       min_vol;
    std::vector<bool>            is_in_batch;
    std::vector<bool>            is_pickable;

public:
    std::vector<std::vector<int>> batches;
    std::vector<double>           squared_distances;

    EllipsoidBatchPicker( const std::vector<Eigen::VectorXd> all_points_input,
                          const std::vector<double>          all_vol_input,
                          const std::vector<Eigen::VectorXd> all_mu_input,
                          const std::vector<Eigen::MatrixXd> all_Sigma_input,
                          const double                       tau_input,    // 3.0 is a good choice
                          const double                       min_vol_rtol) // 1e-5 is a good choice
    {
        tau = tau_input;
        num_pts = all_points_input.size();
        if (all_vol_input.size() != num_pts)
        {
            throw std::invalid_argument( "Different number of points and vol" );
        }
        if (all_mu_input.size() != num_pts)
        {
            throw std::invalid_argument( "Different number of points and mu" );
        }
        if (all_Sigma_input.size() != num_pts)
        {
            throw std::invalid_argument( "Different number of points and Sigma" );
        }
        if ( num_pts == 0 )
        {
            throw std::invalid_argument( "No ellipsoids provided" );
        }
        spatial_dim = all_points_input[0].size();

        squared_distances.reserve(num_pts);
        is_pickable.reserve(num_pts);
        is_in_batch.reserve(num_pts);
        all_points.reserve(num_pts);
        all_vol.reserve(num_pts);
        all_mu.reserve(num_pts);
        all_Sigma.reserve(num_pts);
        for ( int ii=0; ii<num_pts; ++ii )
        {
            all_points.push_back(all_points_input[ii]);
            all_vol.push_back(all_vol_input[ii]);
            all_mu.push_back(all_mu_input[ii]);
            all_Sigma.push_back(all_Sigma_input[ii]);
            is_pickable.push_back(true);
            is_in_batch.push_back(false);
            squared_distances.push_back(-1.0);
        }

        double max_vol = 0.0;
        for ( int ii=0; ii<num_pts; ++ii )
        {
            double vol_i = all_vol[ii];
            if ( vol_i > max_vol )
            {
                max_vol = vol_i;
            }
        }
        min_vol = min_vol_rtol * max_vol;

        Eigen::MatrixXd box_mins(spatial_dim, num_pts);
        Eigen::MatrixXd box_maxes(spatial_dim, num_pts);
        for ( int ii=0; ii<num_pts; ++ii )
        {
            std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(all_mu[ii],
                                                                                    all_Sigma[ii],
                                                                                    tau);
            box_mins.col(ii) = std::get<0>(B);
            box_maxes.col(ii) = std::get<1>(B);
        }
        aabb.build_tree(box_mins, box_maxes);
    }

    std::vector<int> pick_batch()
    {
        for ( int ii=0; ii<num_pts; ++ii )
        {
            is_pickable[ii] = ( (!is_in_batch[ii]) && (all_vol[ii] > min_vol) );
        }

        std::vector<int> candidate_inds(num_pts);
        std::iota(candidate_inds.begin(), candidate_inds.end(), 0);
        stable_sort(candidate_inds.begin(), candidate_inds.end(),
            [this](int i1, int i2) {return squared_distances[i1] > squared_distances[i2];});

        std::vector<int> next_batch;
        for ( int idx : candidate_inds )
        {
            if ( is_pickable[idx] )
            {
                next_batch.push_back(idx);
                is_pickable[idx] = false;
                is_in_batch[idx] = true;
                std::tuple<Eigen::VectorXd, Eigen::VectorXd> B = ellipsoid_bounding_box(all_mu[idx], all_Sigma[idx], tau);
                Eigen::VectorXi possible_collisions = aabb.box_collisions(std::get<0>(B), std::get<1>(B));
                for ( int jj=0; jj<possible_collisions.size(); ++jj )
                {
                    int idx2 = possible_collisions[jj];
                    if ( is_pickable[idx2] )
                    {
                        if ( ellipsoids_intersect(all_mu[idx2], all_Sigma[idx2],
                                                  all_mu[idx],  all_Sigma[idx],
                                                  tau) )
                        {
                            is_pickable[idx2] = false;
                        }
                    }

                }
            }
        }
        batches.push_back(next_batch);

        for ( int ind : next_batch )
        {
            for ( int ii=0; ii<num_pts; ++ii )
            {
                double old_dsq = squared_distances[ii];
                double new_dsq = (all_points[ind] - all_points[ii]).squaredNorm();
                if ( new_dsq < old_dsq || old_dsq < 0.0 )
                {
                    squared_distances[ii] = new_dsq;
                }
            }
        }
        return next_batch;
    }
};

} // end namespace ELLIPSOID