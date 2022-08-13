#pragma once

#include <iostream>
#include <list>
#include <vector>

#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "kdtree.h"
#include "ellipsoid.h"


namespace IMPULSE {

// apply_A    : L2(Omega_in)  -> L2'(Omega_out)
// apply_AT   : L2(Omega_out) -> L2'(Omega_in)
// apply_M_in : L2(Omega_in)  -> L2'(Omega_in)
// solve_M_in : L2'(Omega_in) -> L2(Omega_in)
std::tuple<std::vector<double>,          // all vols, V
           std::vector<Eigen::VectorXd>, // all means, mu
           std::vector<Eigen::MatrixXd>> // all covariances, Sigma
    compute_impulse_response_moments(const std::function<Eigen::VectorXd(Eigen::VectorXd)> & apply_AT,
                                     const std::function<Eigen::VectorXd(Eigen::VectorXd)> & solve_M_in,
                                     const Eigen::MatrixXd & dof_coords_out) // shape=(gdim_out, N_out)
{
    const int d_out = dof_coords_out.rows();
    const int N_out = dof_coords_out.cols();

    std::cout << "computing V" << std::endl;
    Eigen::VectorXd C = Eigen::VectorXd::Ones(N_out);
    Eigen::VectorXd V = solve_M_in(apply_AT(C));
    const int N_in = V.size();

    std::cout << "computing mu" << std::endl;
    Eigen::MatrixXd mu(N_in, d_out);
    for ( int ii=0; ii<d_out; ++ii )
    {
        std::cout << "ii = " << ii << std::endl;
        Eigen::VectorXd L_i = dof_coords_out.row(ii);
        mu.col(ii) = solve_M_in(apply_AT(L_i)).array() / V.array();
    }

    std::cout << "computing Sigma" << std::endl;
    Eigen::MatrixXd Sigma(N_in, d_out*d_out);
    for ( int ii=0; ii<d_out; ++ii )
    {
        Eigen::VectorXd L_i = dof_coords_out.row(ii);
        for ( int jj=0; jj<=ii; ++jj)
        {
            std::cout << "ii = " << ii << ", jj = " << jj << std::endl;
            Eigen::VectorXd L_j = dof_coords_out.row(jj);
            Eigen::VectorXd Q_ij = L_i.array() * L_j.array();
            int ind1 = ii + d_out * jj;
            int ind2 = jj + d_out * ii;
            Sigma.col(ind1) = solve_M_in(apply_AT(Q_ij)).array() / V.array() 
                                - mu.col(ii).array() * mu.col(jj).array();
            if ( ind1 != ind2 )
            {
                Sigma.col(ind2) = Sigma.col(ind1);
            }
        }
    }

    // std::cout << "processing V" << std::endl;
    std::vector<double> all_vol(N_in);
    for ( int ii=0; ii<N_in; ++ii )
    {
        all_vol[ii] = V(ii);
    }

    // std::cout << "processing mu" << std::endl;
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

    // std::cout << "processing Sigma" << std::endl;
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
    // std::cout << "done computing moments" << std::endl;

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
    for ( long unsigned int ii=0; ii<dirac_inds.size(); ++ii )
    {
        dirac_comb(dirac_inds[ii]) = dirac_weights[ii];
    }

    std::cout << "computing impulse response batch" << std::endl;
    return solve_M_out(apply_A(solve_M_in(dirac_comb)));
}

}