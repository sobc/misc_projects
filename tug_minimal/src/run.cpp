#include "run.hpp"

#include "io.hpp"
#include "tug/Boundary.hpp"
#include "tug/Grid.hpp"

#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <vector>

#include <tug/Simulation.hpp>

#include <Eigen/Eigen>

using TugType = double;

using RowMajorMat =
    Eigen::Matrix<TugType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static inline std::vector<TugType>
eigenMatrix_to_vector(const Eigen::MatrixX<TugType> &mat) {
  if (mat.IsRowMajor) {
    return std::vector<TugType>(mat.data(), mat.data() + mat.size());
  } else {
    std::vector<TugType> out_vec(mat.size());
    for (int i = 0; i < mat.rows(); i++) {
      for (int j = 0; j < mat.cols(); j++) {
        out_vec[i * mat.cols() + j] = mat(i, j);
      }
    }
    return out_vec;
  }
}

void run_bench(const bench_input &input, const std::string &output_file,
               const int iterations) {

  std::vector<std::vector<double>> raw_data =
      read_conc_csv<double>(input.csv_file_init, input.ncols, input.nrows);

  std::vector<TugType> raw_alpha_x = read_alpha_csv<TugType>(input.csv_alpha_x);
  std::vector<TugType> raw_alpha_y = read_alpha_csv<TugType>(input.csv_alpha_y);

  Eigen::Map<RowMajorMat> alpha_x(raw_alpha_x.data(), input.nrows, input.ncols);
  Eigen::Map<RowMajorMat> alpha_y(raw_alpha_y.data(), input.nrows, input.ncols);
  // create tug grids and boundary conditions
  for (int i = 0; i < raw_data.size(); i++) {

    Eigen::Map<RowMajorMat> mat(raw_data[i].data(), input.nrows, input.ncols);
    tug::Grid<TugType> grid(input.nrows, input.ncols);

    grid.setConcentrations(mat);
    grid.setDomain(input.s_x, input.s_y);
    grid.setAlpha(alpha_x, alpha_y);

    tug::Boundary<TugType> boundary(grid);

    // set north boundary
    for (const auto &index : input.boundary.north_const) {
      boundary.setBoundaryElementConstant(tug::BC_SIDE_TOP, index,
                                          input.boundary.values[i]);
    }

    // set south boundary
    for (const auto &index : input.boundary.south_const) {
      boundary.setBoundaryElementConstant(tug::BC_SIDE_BOTTOM, index,
                                          input.boundary.values[i]);
    }

    // set east boundary
    for (const auto &index : input.boundary.east_const) {
      boundary.setBoundaryElementConstant(tug::BC_SIDE_RIGHT, index,
                                          input.boundary.values[i]);
    }

    // set west boundary
    for (const auto &index : input.boundary.west_const) {
      boundary.setBoundaryElementConstant(tug::BC_SIDE_LEFT, index,
                                          input.boundary.values[i]);
    }

    tug::Simulation<TugType> sim(grid, boundary);

    sim.setTimestep(input.timestep);
    sim.setIterations(iterations);

    sim.run();

    const auto &result = grid.getConcentrations();

    raw_data[i] = eigenMatrix_to_vector(result);
  }

  // write results to file
  if (!output_file.empty()) {
    write_conc_csv(output_file, raw_data);
  }
}
