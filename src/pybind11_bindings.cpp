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
#include "impulse_response.h"

namespace py = pybind11;

using namespace Eigen;
using namespace std;

using namespace KDT;
using namespace AABB;
using namespace SMESH;
using namespace BRENT;
using namespace ELLIPSOID;
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

    py::class_<EllipsoidBatchPicker>(m, "EllipsoidBatchPicker")
        .def(py::init< const std::vector<Eigen::VectorXd>,
                       const std::vector<Eigen::VectorXd>,
                       const std::vector<Eigen::MatrixXd>,
                       const double>())
        .def_readwrite("batches", &EllipsoidBatchPicker::batches)
        .def("pick_batch", &EllipsoidBatchPicker::pick_batch);

    m.def("impulse_response_moments", &impulse_response_moments);
}

//import numpy as np
//from nalger_helper_functions import brent_minimize
//brent_minimize(lambda x: np.cos(np.exp(x)), -1.0, 1.5, 1e-7, 200)
//Out[3]: (1.1447298817285088, -0.9999999999999999, 12, 13)
