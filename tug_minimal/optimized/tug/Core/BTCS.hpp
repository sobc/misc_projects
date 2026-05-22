/**
 * @file BTCS.hpp
 * @brief Implementation of heterogenous BTCS (backward time-centered space)
 * solution of diffusion equation in 1D and 2D space. Internally the
 * alternating-direction implicit (ADI) method is used. Version 2, because
 * Version 1 was an implementation for the homogeneous BTCS solution.
 *
 */

#ifndef BTCS_H_
#define BTCS_H_

#include "Matrix.hpp"
#include "TugUtils.hpp"

#include <cstddef>
#include <tug/Boundary.hpp>
#include <tug/Grid.hpp>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

namespace tug {

template <class T> class Diagonals {
public:
  Diagonals() : left(), center(), right() {};
  Diagonals(std::size_t size) : left(size), center(size), right(size) {};

public:
  std::vector<T> left;
  std::vector<T> center;
  std::vector<T> right;
};

// calculates coefficient for boundary in constant case
template <class T>
constexpr std::pair<T, T> calcBoundaryCoeffConstant(T alpha_center,
                                                    T alpha_side, T sx) {
  const T centerCoeff = 1 + sx * (calcAlphaIntercell(alpha_center, alpha_side) +
                                  2 * alpha_center);
  const T sideCoeff = -sx * calcAlphaIntercell(alpha_center, alpha_side);

  return {centerCoeff, sideCoeff};
}

// calculates coefficient for boundary in closed case
template <class T>
constexpr std::pair<T, T> calcBoundaryCoeffClosed(T alpha_center, T alpha_side,
                                                  T sx) {
  const T centerCoeff = 1 + sx * calcAlphaIntercell(alpha_center, alpha_side);

  const T sideCoeff = -sx * calcAlphaIntercell(alpha_center, alpha_side);

  return {centerCoeff, sideCoeff};
}

// creates coefficient matrix for next time step from alphas in x-direction
template <class T>
static Diagonals<T>
createCoeffMatrix(const RowMajMat<T> &alpha,
                  const std::vector<BoundaryElement<T>> &bcLeft,
                  const std::vector<BoundaryElement<T>> &bcRight, int numCols,
                  int rowIndex, T sx) {
  // square matrix of column^2 dimension for the coefficients
  Diagonals<T> diag(numCols);

  // left column
  {
    switch (bcLeft[rowIndex].getType()) {
    case BC_TYPE_CONSTANT: {
      auto [centerCoeffTop, rightCoeffTop] =
          calcBoundaryCoeffConstant(alpha(rowIndex, 0), alpha(rowIndex, 1), sx);
      diag.center[0] = centerCoeffTop;
      diag.right[0] = rightCoeffTop;
      break;
    }
    case BC_TYPE_CLOSED: {
      auto [centerCoeffTop, rightCoeffTop] =
          calcBoundaryCoeffClosed(alpha(rowIndex, 0), alpha(rowIndex, 1), sx);
      diag.center[0] = centerCoeffTop;
      diag.right[0] = rightCoeffTop;
      break;
    }
    default: {
      throw_invalid_argument(
          "Undefined Boundary Condition Type somewhere on Left or Top!");
    }
    }
  }

  // inner columns
  int n = numCols - 1;
  for (int i = 1; i < n; i++) {
    diag.left[i] =
        -sx * calcAlphaIntercell(alpha(rowIndex, i - 1), alpha(rowIndex, i));
    diag.center[i] =
        1 +
        sx * (calcAlphaIntercell(alpha(rowIndex, i), alpha(rowIndex, i + 1)) +
              calcAlphaIntercell(alpha(rowIndex, i - 1), alpha(rowIndex, i)));
    diag.right[i] =
        -sx * calcAlphaIntercell(alpha(rowIndex, i), alpha(rowIndex, i + 1));
  }

  // right column
  {
    switch (bcRight[rowIndex].getType()) {
    case BC_TYPE_CONSTANT: {
      auto [centerCoeffBottom, leftCoeffBottom] = calcBoundaryCoeffConstant(
          alpha(rowIndex, n), alpha(rowIndex, n - 1), sx);
      diag.left[n] = leftCoeffBottom;
      diag.center[n] = centerCoeffBottom;
      break;
    }
    case BC_TYPE_CLOSED: {
      auto [centerCoeffBottom, leftCoeffBottom] = calcBoundaryCoeffClosed(
          alpha(rowIndex, n), alpha(rowIndex, n - 1), sx);
      diag.left[n] = leftCoeffBottom;
      diag.center[n] = centerCoeffBottom;
      break;
    }
    default: {
      throw_invalid_argument(
          "Undefined Boundary Condition Type somewhere on Right or Bottom!");
    }
    }
  }

  return diag;
}

// calculates explicit concentration at boundary in closed case
template <typename T>
constexpr T calcExplicitConcentrationsBoundaryClosed(T conc_center,
                                                     T alpha_center,
                                                     T alpha_neigbor, T sy) {
  return sy * calcAlphaIntercell(alpha_center, alpha_neigbor) * conc_center +
         (1 - sy * (calcAlphaIntercell(alpha_center, alpha_neigbor))) *
             conc_center;
}

// calculates explicity concentration at boundary in constant case
template <typename T>
constexpr T calcExplicitConcentrationsBoundaryConstant(T conc_center, T conc_bc,
                                                       T alpha_center,
                                                       T alpha_neighbor, T sy) {
  const T inter_cell = calcAlphaIntercell(alpha_center, alpha_neighbor);
  return sy * inter_cell * conc_center +
         (1 - sy * (inter_cell + alpha_center)) * conc_center +
         sy * alpha_center * conc_bc;
}

// creates a solution vector for next time step from the current state of
// concentrations
template <class T>
static Eigen::VectorX<T>
createSolutionVector(const RowMajMat<T> &concentrations,
                     const RowMajMat<T> &alphaX, const RowMajMat<T> &alphaY,
                     const std::vector<BoundaryElement<T>> &bcLeft,
                     const std::vector<BoundaryElement<T>> &bcRight,
                     const std::vector<BoundaryElement<T>> &bcTop,
                     const std::vector<BoundaryElement<T>> &bcBottom,
                     int length, int rowIndex, T sx, T sy) {
  Eigen::VectorX<T> sv(length);
  const std::size_t numRows = concentrations.rows();

  // inner rows
  if (rowIndex > 0 && rowIndex < numRows - 1) {
    for (int i = 0; i < length; i++) {
      sv(i) =
          sy *
              calcAlphaIntercell(alphaY(rowIndex, i), alphaY(rowIndex + 1, i)) *
              concentrations(rowIndex + 1, i) +
          (1 - sy * (calcAlphaIntercell(alphaY(rowIndex, i),
                                        alphaY(rowIndex + 1, i)) +
                     calcAlphaIntercell(alphaY(rowIndex - 1, i),
                                        alphaY(rowIndex, i)))) *
              concentrations(rowIndex, i) +
          sy *
              calcAlphaIntercell(alphaY(rowIndex - 1, i), alphaY(rowIndex, i)) *
              concentrations(rowIndex - 1, i);
    }
  }

  // first row
  else if (rowIndex == 0) {
    for (int i = 0; i < length; i++) {
      switch (bcTop[i].getType()) {
      case BC_TYPE_CONSTANT: {
        sv(i) = calcExplicitConcentrationsBoundaryConstant(
            concentrations(rowIndex, i), bcTop[i].getValue(),
            alphaY(rowIndex, i), alphaY(rowIndex + 1, i), sy);
        break;
      }
      case BC_TYPE_CLOSED: {
        sv(i) = calcExplicitConcentrationsBoundaryClosed(
            concentrations(rowIndex, i), alphaY(rowIndex, i),
            alphaY(rowIndex + 1, i), sy);
        break;
      }
      default:
        throw_invalid_argument("Undefined Boundary Condition Type "
                               "somewhere on Left or Top!");
      }
    }
  }

  // last row
  else if (rowIndex == numRows - 1) {
    for (int i = 0; i < length; i++) {
      switch (bcBottom[i].getType()) {
      case BC_TYPE_CONSTANT: {
        sv(i) = calcExplicitConcentrationsBoundaryConstant(
            concentrations(rowIndex, i), bcBottom[i].getValue(),
            alphaY(rowIndex, i), alphaY(rowIndex - 1, i), sy);
        break;
      }
      case BC_TYPE_CLOSED: {
        sv(i) = calcExplicitConcentrationsBoundaryClosed(
            concentrations(rowIndex, i), alphaY(rowIndex, i),
            alphaY(rowIndex - 1, i), sy);
        break;
      }
      default:
        throw_invalid_argument("Undefined Boundary Condition Type "
                               "somewhere on Right or Bottom!");
      }
    }
  }

  // first column -> additional fixed concentration change from perpendicular
  // dimension in constant bc case
  if (bcLeft[rowIndex].getType() == BC_TYPE_CONSTANT) {
    sv(0) += 2 * sx * alphaX(rowIndex, 0) * bcLeft[rowIndex].getValue();
  }

  // last column -> additional fixed concentration change from perpendicular
  // dimension in constant bc case
  if (bcRight[rowIndex].getType() == BC_TYPE_CONSTANT) {
    sv(length - 1) +=
        2 * sx * alphaX(rowIndex, length - 1) * bcRight[rowIndex].getValue();
  }

  return sv;
}

// solver for linear equation system; A corresponds to coefficient matrix,
// b to the solution vector
// implementation of Thomas Algorithm
template <class T>
static Eigen::VectorX<T> ThomasAlgorithm(Diagonals<T> &A,
                                         Eigen::VectorX<T> &b) {
  Eigen::Index n = b.size();
  Eigen::VectorX<T> x_vec = b;

  // HACK: write CSV to file
#ifdef WRITE_THOMAS_CSV
#include <fstream>
#include <string>
  static std::uint32_t file_index = 0;
  std::string file_name = "Thomas_" + std::to_string(file_index++) + ".csv";

  std::ofstream out_file;

  out_file.open(file_name, std::ofstream::trunc | std::ofstream::out);

  // print header
  out_file << "Aa, Ab, Ac, b\n";

  // iterate through all elements
  for (std::size_t i = 0; i < n; i++) {
    out_file << A.left[i] << ", " << A.center[i] << ", " << A.right[i] << ", "
             << b[i] << "\n";
  }

  out_file.close();
#endif

  // start solving - c_diag and x_vec are overwritten
  n--;
  A.right[0] /= A.center[0];
  x_vec[0] /= A.center[0];

  for (Eigen::Index i = 1; i < n; i++) {
    A.right[i] /= A.center[i] - A.left[i] * A.right[i - 1];
    x_vec[i] = (x_vec[i] - A.left[i] * x_vec[i - 1]) /
               (A.center[i] - A.left[i] * A.right[i - 1]);
  }

  x_vec[n] = (x_vec[n] - A.left[n] * x_vec[n - 1]) /
             (A.center[n] - A.left[n] * A.right[n - 1]);

  for (Eigen::Index i = n; i-- > 0;) {
    x_vec[i] -= A.right[i] * x_vec[i + 1];
  }

  return x_vec;
}

// BTCS solution for 2D grid
template <class T>
static void BTCS_2D(Grid<T> &grid, Boundary<T> &bc, T timestep,
                    int iterations) {
  int rowMax = grid.getRow();
  int colMax = grid.getCol();
  T sx = timestep / (2 * grid.getDeltaCol() * grid.getDeltaCol());
  T sy = timestep / (2 * grid.getDeltaRow() * grid.getDeltaRow());

  RowMajMat<T> concentrations_t1(rowMax, colMax);

  const auto &bcLeft = bc.getBoundarySide(BC_SIDE_LEFT);
  const auto &bcRight = bc.getBoundarySide(BC_SIDE_RIGHT);
  const auto &bcTop = bc.getBoundarySide(BC_SIDE_TOP);
  const auto &bcBottom = bc.getBoundarySide(BC_SIDE_BOTTOM);

  RowMajMat<T> &concentrations = grid.getConcentrations();
  const RowMajMat<T> &alphaX = grid.getAlphaX();
  const RowMajMat<T> &alphaY = grid.getAlphaY();
  const RowMajMat<T> alphaX_t = alphaX.transpose();
  const RowMajMat<T> alphaY_t = alphaY.transpose();

  for (int iter = 0; iter < iterations; iter++) {

#pragma omp parallel for
    for (int i = 0; i < rowMax; i++) {
      Diagonals<T> A =
          createCoeffMatrix(alphaX, bcLeft, bcRight, colMax, i, sx);
      Eigen::VectorX<T> b =
          createSolutionVector(concentrations, alphaX, alphaY, bcLeft, bcRight,
                               bcTop, bcBottom, colMax, i, sx, sy);

      concentrations_t1.row(i) = ThomasAlgorithm(A, b);
    }

    concentrations_t1.transposeInPlace();

#pragma omp parallel for
    for (int i = 0; i < colMax; i++) {
      // swap alphas, boundary conditions and sx/sy for column-wise
      // calculation
      Diagonals<T> A =
          createCoeffMatrix(alphaY_t, bcTop, bcBottom, rowMax, i, sy);
      Eigen::VectorX<T> b =
          createSolutionVector(concentrations_t1, alphaY_t, alphaX_t, bcTop,
                               bcBottom, bcLeft, bcRight, rowMax, i, sy, sx);

      concentrations.col(i) = ThomasAlgorithm(A, b);
    }
  }
}

// entry point for Thomas algorithm solver; differentiate 1D and 2D grid
template <class T>
void BTCS_Thomas(Grid<T> &grid, Boundary<T> &bc, T timestep, int iterations) {
  if (grid.getDim() == 2) {
    BTCS_2D(grid, bc, timestep, iterations);
  } else {
    throw_invalid_argument("Error: Only 2-dimensional grids are defined!");
  }
}
} // namespace tug

#endif // BTCS_H_
