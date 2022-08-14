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

#include "kdtree.h"
#include "aabbtree.h"
#include "simplexmesh.h"
#include "brent_minimize.h"
#include "ellipsoid.h"
#include "interpolation.h"
#include "impulse_response.h"
#include "product_convolution_kernel.h"

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


PYBIND11_MODULE(localpsfcpp, m) {
    m.doc() = "python bindings for localpsf c++ code";

    py::class_<KDTree>(m, "KDTree")
        .def(py::init< const Ref<const MatrixXd> >())
        .def_readwrite("block_size", &KDTree::block_size)
        .def("query", &KDTree::query, "many querys, many neighbor");

    py::class_<AABBTree>(m, "AABBTree")
        .def(py::init< const Ref<const MatrixXd>,
                       const Ref<const MatrixXd> >())
        .def("point_collisions", &AABBTree::point_collisions)
        .def("point_collisions_vectorized", &AABBTree::point_collisions_vectorized)
        .def("box_collisions", &AABBTree::box_collisions)
        .def("box_collisions_vectorized", &AABBTree::box_collisions_vectorized)
        .def("ball_collisions", &AABBTree::ball_collisions)
        .def("ball_collisions_vectorized", &AABBTree::ball_collisions_vectorized);

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

    m.def("TPS_interpolate", &TPS_interpolate);
    m.def("LMDI_points_and_values", &LMDI_points_and_values);

    py::enum_<ShiftMethod>(m, "ShiftMethod")
        .value("LOW_RANK",                           ShiftMethod::LOW_RANK)
        .value("LOCAL_TRANSLATION_INVARIANCE",       ShiftMethod::LOCAL_TRANSLATION_INVARIANCE)
        .value("LOCAL_MEAN_DISPLACEMENT_INVARIANCE", ShiftMethod::LOCAL_MEAN_DISPLACEMENT_INVARIANCE)
        .value("ELLIPSOID_MAPPING",                  ShiftMethod::ELLIPSOID_MAPPING);

    py::enum_<ScalingMethod>(m, "ScalingMethod")
        .value("NONE",    ScalingMethod::NONE)
        .value("VOL",     ScalingMethod::VOL)
        .value("DET",     ScalingMethod::DET)
        .value("DETVOL", ScalingMethod::DETVOL);

    py::class_<LPSFKernel>(m, "LPSFKernel")
        .def("add_batch", &LPSFKernel::add_batch)
        .def("entry",     &LPSFKernel::entry)
        .def("block",     &LPSFKernel::block)
        .def_readonly("dS", &LPSFKernel::dS)
        .def_readonly("dT", &LPSFKernel::dT)
        .def_readonly("NS", &LPSFKernel::NS)
        .def_readonly("NT", &LPSFKernel::NT)
        .def_readonly("source_vertices", &LPSFKernel::source_vertices)
        .def_readonly("target_vertices", &LPSFKernel::target_vertices)
        .def_readonly("target_mesh", &LPSFKernel::target_mesh)
        .def_readonly("apply_A",     &LPSFKernel::apply_A)
        .def_readonly("apply_AT",    &LPSFKernel::apply_AT)
        .def_readonly("apply_M_in",  &LPSFKernel::apply_M_in)
        .def_readonly("apply_M_out", &LPSFKernel::apply_M_out)
        .def_readonly("solve_M_in",  &LPSFKernel::solve_M_in)
        .def_readonly("solve_M_out", &LPSFKernel::solve_M_out)
        .def_readonly("vol",   &LPSFKernel::vol)
        .def_readonly("mu",    &LPSFKernel::mu)
        .def_readonly("Sigma", &LPSFKernel::Sigma)
        .def_readonly("tau",   &LPSFKernel::tau)
        .def_readonly("Sigma_is_good",  &LPSFKernel::Sigma_is_good)
        .def_readonly("inv_Sigma",      &LPSFKernel::inv_Sigma)
        .def_readonly("sqrt_Sigma",     &LPSFKernel::sqrt_Sigma)
        .def_readonly("inv_sqrt_Sigma", &LPSFKernel::inv_sqrt_Sigma)
        .def_readonly("det_sqrt_Sigma", &LPSFKernel::det_sqrt_Sigma)
        .def_readonly("ellipsoid_aabb", &LPSFKernel::ellipsoid_aabb)
        .def_readonly("min_vol_rtol",   &LPSFKernel::min_vol_rtol)
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

}


//import numpy as np
//from nalger_helper_functions import brent_minimize
//brent_minimize(lambda x: np.cos(np.exp(x)), -1.0, 1.5, 1e-7, 200)
//Out[3]: (1.1447298817285088, -0.9999999999999999, 12, 13)
