#include <algorithm>
//#include <chrono>
#include <pybind11/pybind11.h>
#include <iostream>
#include <exception>
#include "linear_regression.h"

using namespace std;
namespace py=pybind11;

LinearRegression::LinearRegression()
{
}

void LinearRegression::fit(Matrix X, Matrix y)
{
    // Aplicar LDLT para ver si va mas rapido usando cholesovbksy
    _solucion = (X.transpose() * X).completeOrthogonalDecomposition().solve(X.transpose() * y);
}


Matrix LinearRegression::predict(Matrix X)
{
    return X * _solucion;
}
