#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

#include <Eigen/Sparse>

// Define FrameDeltaSparseMatrices as a struct containing sparse matrices
struct FrameDeltaSparseMatrices {
    Eigen::SparseMatrix<double, Eigen::RowMajor> Delta_A_L_B_L;
    Eigen::SparseMatrix<double, Eigen::RowMajor> Delta_A_L_B_R;
    Eigen::SparseMatrix<double, Eigen::RowMajor> Delta_A_R_B_L;
    Eigen::SparseMatrix<double, Eigen::RowMajor> Delta_A_R_B_R;

    // Optional: Constructor to initialize the sparse matrices
    FrameDeltaSparseMatrices(int rows = 0, int cols = 0)
        : Delta_A_L_B_L(rows, cols), Delta_A_L_B_R(rows, cols),
          Delta_A_R_B_L(rows, cols), Delta_A_R_B_R(rows, cols) {}
};

#endif // DATASTRUCTURES_H
