#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <Eigen/LU>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <hlib.hh>

#include "kdtree.h"
#include "aabbtree.h"
#include "simplexmesh.h"
#include "brent_minimize.h"
#include "ellipsoid.h"
#include "interpolation.h"
#include "impulse_response.h"
#include "product_convolution_kernel.h"
#include "hmatrix.h"
#include "lpsf_utils.h"

namespace py = pybind11;

using namespace Eigen;
using namespace std;

using namespace KDT;
using namespace AABB;
using namespace SMESH;
using namespace BRENT;
using namespace ELLIPSOID;
using namespace INTERP;
using namespace IMPULSE;
using namespace PCK;
using namespace HMAT;
using namespace LPSFUTIL;


PYBIND11_MODULE(localpsfcpp, m) {
    m.doc() = "python bindings for localpsf c++ code";


    // kdtree.h
    py::class_<KDTree>(m, "KDTree")
        .def(py::init< const Ref<const MatrixXd> >())
        .def_readwrite("block_size", &KDTree::block_size)
        .def("query", &KDTree::query, "many querys, many neighbor");


    // aabbtree.h
    py::class_<AABBTree>(m, "AABBTree")
        .def(py::init< const Ref<const MatrixXd>,
                       const Ref<const MatrixXd> >())
        .def("point_collisions", &AABBTree::point_collisions)
        .def("point_collisions_vectorized", &AABBTree::point_collisions_vectorized)
        .def("box_collisions", &AABBTree::box_collisions)
        .def("box_collisions_vectorized", &AABBTree::box_collisions_vectorized)
        .def("ball_collisions", &AABBTree::ball_collisions)
        .def("ball_collisions_vectorized", &AABBTree::ball_collisions_vectorized);


    // simplexmesh.h
    py::class_<SimplexMesh>(m, "SimplexMesh")
        .def(py::init< const Ref<const MatrixXd>,
                       const Ref<const MatrixXi> >())
        .def("closest_point", &SimplexMesh::closest_point)
        .def("point_is_in_mesh", &SimplexMesh::point_is_in_mesh)
        .def("first_point_collision", &SimplexMesh::first_point_collision)
        .def("eval_CG1", &SimplexMesh::eval_CG1);

    m.def("brent_minimize", &brent_minimize);
    m.def("ellipsoids_intersect", &ellipsoids_intersect);

    m.def("compute_impulse_response_moments", &compute_impulse_response_moments);
    m.def("compute_impulse_response_batch", &compute_impulse_response_batch);
    m.def("make_ellipsoid_aabbtree", &make_ellipsoid_aabbtree);
    m.def("pick_ellipsoid_batch", &pick_ellipsoid_batch);


    // interpolation.h
    m.def("RBF_TPS_interpolate", &RBF_TPS_interpolate);
    m.def("RBF_GAUSS_interpolate", &RBF_GAUSS_interpolate);
    m.def("LMDI_points_and_values", &LMDI_points_and_values);

    py::enum_<ShiftMethod>(m, "ShiftMethod")
        .value("LOW_RANK",                           ShiftMethod::LOW_RANK)
        .value("LOCAL_TRANSLATION_INVARIANCE",       ShiftMethod::LOCAL_TRANSLATION_INVARIANCE)
        .value("LOCAL_MEAN_DISPLACEMENT_INVARIANCE", ShiftMethod::LOCAL_MEAN_DISPLACEMENT_INVARIANCE)
        .value("ELLIPSOID_MAPPING",                  ShiftMethod::ELLIPSOID_MAPPING);

    py::enum_<ScalingMethod>(m, "ScalingMethod")
        .value("NONE",   ScalingMethod::NONE)
        .value("VOL",    ScalingMethod::VOL)
        .value("DET",    ScalingMethod::DET)
        .value("DETVOL", ScalingMethod::DETVOL);

    py::enum_<InterpolationMethod>(m, "InterpolationMethod")
        .value("RBF_THIN_PLATE_SPLINES", InterpolationMethod::RBF_THIN_PLATE_SPLINES)
        .value("RBF_GAUSS",              InterpolationMethod::RBF_GAUSS);


    // product_convolution_kernel.h
    py::class_<LPSFKernel, std::shared_ptr<LPSFKernel>>(m, "LPSFKernel")
        .def("num_batches", &LPSFKernel::num_batches)
        .def("add_batch",   &LPSFKernel::add_batch)
        .def("entry",       &LPSFKernel::entry)
        .def("block",       &LPSFKernel::block)
        .def_readwrite("dS", &LPSFKernel::dS)
        .def_readwrite("dT", &LPSFKernel::dT)
        .def_readwrite("NS", &LPSFKernel::NS)
        .def_readwrite("NT", &LPSFKernel::NT)
        .def_readwrite("source_vertices", &LPSFKernel::source_vertices)
        .def_readwrite("target_vertices", &LPSFKernel::target_vertices)
        .def_readwrite("target_mesh", &LPSFKernel::target_mesh)
        .def_readwrite("apply_A",     &LPSFKernel::apply_A)
        .def_readwrite("apply_AT",    &LPSFKernel::apply_AT)
        .def_readwrite("apply_M_in",  &LPSFKernel::apply_M_in)
        .def_readwrite("apply_M_out", &LPSFKernel::apply_M_out)
        .def_readwrite("solve_M_in",  &LPSFKernel::solve_M_in)
        .def_readwrite("solve_M_out", &LPSFKernel::solve_M_out)
        .def_readwrite("vol",   &LPSFKernel::vol)
        .def_readwrite("mu",    &LPSFKernel::mu)
        .def_readwrite("Sigma", &LPSFKernel::Sigma)
        .def_readwrite("tau",   &LPSFKernel::tau)
        .def_readwrite("Sigma_is_good",  &LPSFKernel::Sigma_is_good)
        .def_readwrite("inv_Sigma",      &LPSFKernel::inv_Sigma)
        .def_readwrite("sqrt_Sigma",     &LPSFKernel::sqrt_Sigma)
        .def_readwrite("inv_sqrt_Sigma", &LPSFKernel::inv_sqrt_Sigma)
        .def_readwrite("det_sqrt_Sigma", &LPSFKernel::det_sqrt_Sigma)
        .def_readwrite("ellipsoid_aabb", &LPSFKernel::ellipsoid_aabb)
        .def_readwrite("min_vol_rtol",   &LPSFKernel::min_vol_rtol)
        .def_readwrite("eta_batches",             &LPSFKernel::eta_batches)
        .def_readwrite("dirac_ind_batches",       &LPSFKernel::dirac_ind_batches)
        .def_readwrite("dirac_squared_distances", &LPSFKernel::dirac_squared_distances)
        .def_readwrite("dirac_inds",              &LPSFKernel::dirac_inds)
        .def_readwrite("dirac_points",            &LPSFKernel::dirac_points)
        .def_readwrite("dirac_weights",           &LPSFKernel::dirac_weights)
        .def_readwrite("dirac2batch",             &LPSFKernel::dirac2batch)
        .def_readwrite("dirac_kdtree",            &LPSFKernel::dirac_kdtree)
        .def_readwrite("num_neighbors", &LPSFKernel::num_neighbors);

    m.def("create_LPSFKernel", &create_LPSFKernel);


    // hmatrix.h
    m.def("build_lpsfkernel_hmatrix", &build_lpsfkernel_hmatrix);


    // HLIBPro
    py::enum_<HLIB::matop_t>(m, "matop_t")
        .value("MATOP_NORM",       HLIB::matop_t::MATOP_NORM)
        .value("apply_normal",     HLIB::matop_t::apply_normal)
        .value("MATOP_TRANS",      HLIB::matop_t::MATOP_TRANS)
        .value("apply_trans",      HLIB::matop_t::apply_trans)
        .value("apply_transposed", HLIB::matop_t::apply_transposed)
        .value("MATOP_ADJ",        HLIB::matop_t::MATOP_ADJ)
        .value("MATOP_CONJTRANS",  HLIB::matop_t::MATOP_CONJTRANS)
        .value("apply_adj",        HLIB::matop_t::apply_adj)
        .value("apply_adjoint",    HLIB::matop_t::apply_adjoint)
        .value("apply_conjtrans",  HLIB::matop_t::apply_conjtrans)
        .export_values();

    py::class_<HLIB::TMatrix, std::shared_ptr<HLIB::TMatrix>>(m, "TMatrix")
        .def("id", &HLIB::TMatrix::id)
        .def("rows", &HLIB::TMatrix::rows)
        .def("cols", &HLIB::TMatrix::cols)
        .def("is_nonsym", &HLIB::TMatrix::is_nonsym)
        .def("is_symmetric", &HLIB::TMatrix::is_symmetric)
        .def("is_hermitian", &HLIB::TMatrix::is_hermitian)
        .def("set_nonsym", &HLIB::TMatrix::set_nonsym)
        .def("is_real", &HLIB::TMatrix::is_real)
        .def("is_complex", &HLIB::TMatrix::is_complex)
        .def("to_real", &HLIB::TMatrix::to_real)
        .def("to_complex", &HLIB::TMatrix::to_complex)
        .def("add_update", &HLIB::TMatrix::add_update)
        .def("entry", &HLIB::TMatrix::entry)
        // .def("apply", &HLIB::TMatrix::apply, py::arg("x"), py::arg("y"), py::arg("op")=HLIB::apply_normal)
        .def("set_symmetric", &HLIB::TMatrix::set_symmetric)
        .def("set_hermitian", &HLIB::TMatrix::set_hermitian)
        .def("domain_dim", &HLIB::TMatrix::domain_dim)
        .def("range_dim", &HLIB::TMatrix::range_dim)
        // .def("domain_vector", &HLIB::TMatrix::domain_vector)
        // .def("range_vector", &HLIB::TMatrix::range_vector)
        .def("transpose", &HLIB::TMatrix::transpose)
        .def("conjugate", &HLIB::TMatrix::conjugate)
        .def("add", &HLIB::TMatrix::add)
        .def("scale", &HLIB::TMatrix::scale)
        .def("truncate", &HLIB::TMatrix::truncate)
        // .def("mul_vec", &HLIB::TMatrix::mul_vec, py::arg("alpha"), py::arg("x"),
        //                                          py::arg("beta"), py::arg("y"), py::arg("op")=HLIB::MATOP_NORM)
        .def("check_data", &HLIB::TMatrix::check_data)
        .def("byte_size", &HLIB::TMatrix::byte_size)
        .def("print", &HLIB::TMatrix::print)
        .def("create", &HLIB::TMatrix::create)
        .def("cluster", &HLIB::TMatrix::cluster)
        // .def("row_vector", &HLIB::TMatrix::row_vector)
        // .def("col_vector", &HLIB::TMatrix::col_vector)
        // .def("form", &HLIB::TMatrix::form)
        ;

    py::class_<HLIB::TClusterTree, std::shared_ptr<HLIB::TClusterTree>>(m, "TClusterTree")
        // .def("perm_i2e", &HLIB::TClusterTree::perm_i2e)
        // .def("perm_e2i", &HLIB::TClusterTree::perm_e2i)
        .def("nnodes", &HLIB::TClusterTree::nnodes)
        .def("depth", &HLIB::TClusterTree::depth)
        .def("byte_size", &HLIB::TClusterTree::byte_size);

    py::class_<HLIB::TBlockClusterTree, std::shared_ptr<HLIB::TBlockClusterTree>>(m, "TBlockClusterTree")
        .def("row_ct", &HLIB::TBlockClusterTree::row_ct)
        .def("col_ct", &HLIB::TBlockClusterTree::col_ct)
        .def("nnodes", &HLIB::TBlockClusterTree::nnodes)
        .def("depth", &HLIB::TBlockClusterTree::depth)
        .def("nnodes", &HLIB::TBlockClusterTree::nnodes)
        .def("depth", &HLIB::TBlockClusterTree::depth)
        .def("compute_c_sp", &HLIB::TBlockClusterTree::compute_c_sp)
        .def("byte_size", &HLIB::TBlockClusterTree::byte_size);

    py::class_<HLIB::TTruncAcc>(m, "TTruncAcc")
        .def("max_rank", &HLIB::TTruncAcc::max_rank)
        .def("has_max_rank", &HLIB::TTruncAcc::has_max_rank)
        .def("rel_eps", &HLIB::TTruncAcc::rel_eps)
        .def("abs_eps", &HLIB::TTruncAcc::abs_eps)
        .def(py::init<>())
//        .def(py::init<const int, double>(), py::arg("k"), py::arg("absolute_eps")=CFG::Arith::abs_eps)
        .def(py::init<const double, double>(), py::arg("relative_eps"), py::arg("absolute_eps")=HLIB::CFG::Arith::abs_eps);

    m.def("build_lpsfkernel_hmatrix", &build_lpsfkernel_hmatrix);

    m.def("build_cluster_tree_from_dof_coords", &build_cluster_tree_from_dof_coords);
    m.def("build_block_cluster_tree", &build_block_cluster_tree);
    m.def("visualize_cluster_tree", &visualize_cluster_tree);
    m.def("visualize_block_cluster_tree", &visualize_block_cluster_tree);
    m.def("visualize_hmatrix", &visualize_hmatrix);
    m.def("TMatrix_entry", &TMatrix_entry);
    m.def("TMatrix_submatrix", &TMatrix_submatrix);

    // lpsf_utils.h
    m.def("make_unit_square_mesh", &make_unit_square_mesh);

}


//import numpy as np
//from nalger_helper_functions import brent_minimize
//brent_minimize(lambda x: np.cos(np.exp(x)), -1.0, 1.5, 1e-7, 200)
//Out[3]: (1.1447298817285088, -0.9999999999999999, 12, 13)
