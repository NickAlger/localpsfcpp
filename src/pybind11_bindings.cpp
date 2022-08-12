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

namespace py = pybind11;

using namespace Eigen;
using namespace std;

using namespace KDT;
using namespace AABB;
using namespace SMESH;
using namespace BRENT;
using namespace ELLIPSOID;
using namespace INTERP;
using namespace IMPULSERESPONSE;


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

    // py::class_<EllipsoidBatchPicker>(m, "EllipsoidBatchPicker")
    //     .def(py::init< const std::vector<Eigen::VectorXd>,
    //                    const std::vector<double>,
    //                    const std::vector<Eigen::VectorXd>,
    //                    const std::vector<Eigen::MatrixXd>,
    //                    const double,
    //                    const double>())
    //     .def_readwrite("batches", &EllipsoidBatchPicker::batches)
    //     .def("pick_batch", &EllipsoidBatchPicker::pick_batch);

    // py::class_<EllipsoidForest>(m, "EllipsoidForest")
    //     .def(py::init< 
    //     // const std::vector<Eigen::VectorXd> &, // reference_points_list,
    //                    const std::vector<double>          &, // vol_list,
    //                    const std::vector<Eigen::VectorXd> &, // mu_list,
    //                    const std::vector<Eigen::MatrixXd> &, // Sigma_list,
    //                    const double                          // initial_tau 
    //                  >())
    //     .def_readwrite("vol",                &EllipsoidForest::vol)
    //     .def_readwrite("mu",                 &EllipsoidForest::mu)
    //     .def_readwrite("Sigma",              &EllipsoidForest::Sigma)
    //     .def_readwrite("tau",                &EllipsoidForest::tau)
    //     .def_readwrite("d",                  &EllipsoidForest::d)
    //     .def_readwrite("N",                  &EllipsoidForest::N)
    //     .def_readwrite("Sigma_eigenvectors", &EllipsoidForest::Sigma_eigenvectors)
    //     .def_readwrite("Sigma_eigenvalues",  &EllipsoidForest::Sigma_eigenvalues)
    //     .def_readwrite("iSigma",             &EllipsoidForest::iSigma)
    //     .def_readwrite("sqrt_Sigma",         &EllipsoidForest::sqrt_Sigma)
    //     .def_readwrite("isqrt_Sigma",        &EllipsoidForest::isqrt_Sigma)
    //     .def_readwrite("det_sqrt_Sigma",     &EllipsoidForest::det_sqrt_Sigma)
    //     .def_readwrite("box_mins",           &EllipsoidForest::box_mins)
    //     .def_readwrite("box_maxes",          &EllipsoidForest::box_maxes)
    //     .def_readwrite("ellipsoid_aabb",     &EllipsoidForest::ellipsoid_aabb)
    //     .def("update_tau", &EllipsoidForest::update_tau)
    //     .def("pick_ellipsoid_batch", &EllipsoidForest::pick_ellipsoid_batch);

    m.def("compute_impulse_response_moments", &compute_impulse_response_moments);
    m.def("compute_impulse_response_batch", &compute_impulse_response_batch);
    m.def("make_ellipsoid_aabbtree", &make_ellipsoid_aabbtree);
    m.def("pick_ellipsoid_batch", &pick_ellipsoid_batch);

    m.def("TPS_interpolate", &TPS_interpolate);
    m.def("LMDI_points_and_values", &LMDI_points_and_values);

}


//import numpy as np
//from nalger_helper_functions import brent_minimize
//brent_minimize(lambda x: np.cos(np.exp(x)), -1.0, 1.5, 1e-7, 200)
//Out[3]: (1.1447298817285088, -0.9999999999999999, 12, 13)
