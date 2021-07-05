#include <algorithm>
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
    _solucion = (X.transpose() * X).completeOrthogonalDecomposition().solve(X.transpose() * y);
}


Matrix LinearRegression::predict(Matrix X)
{
    return X * _solucion;
}
