#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "linear_regression.h"

namespace py=pybind11;

// el primer argumento es el nombre...
PYBIND11_MODULE(metnum, m) {
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit", &LinearRegression::fit)
        .def("fitLDLT", &LinearRegression::fitLDLT)
        .def("fitLLT", &LinearRegression::fitLLT)
        .def("fitHouseholderQR", &LinearRegression::fitHouseholderQR)
        .def("fitFullPivHouseholderQR", &LinearRegression::fitFullPivHouseholderQR)
        .def("predict", &LinearRegression::predict);
}
