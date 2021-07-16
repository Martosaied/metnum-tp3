#pragma once

#include "types.h"

class LinearRegression {
public:
    LinearRegression();

    void fit(Matrix X, Matrix y);

    void fitLDLT(Matrix X, Matrix y);
    
    void fitLLT(Matrix X, Matrix y);

    void fitHouseholderQR(Matrix X, Matrix y);

    void fitFullPivHouseholderQR(Matrix X, Matrix y);

    Matrix predict(Matrix X);
private:
    Matrix _solucion;
};
