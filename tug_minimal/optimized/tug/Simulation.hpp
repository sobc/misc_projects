/**
 * @file Simulation.hpp
 * @brief API of Simulation class, that holds all information regarding a
 * specific simulation run like its timestep, number of iterations and output
 * options. Simulation object also holds a predefined Grid and Boundary object.
 *
 */

#ifndef SIMULATION_H_
#define SIMULATION_H_

#include "Boundary.hpp"
#include "Grid.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Core/BTCS.hpp"
#include "Core/TugUtils.hpp"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_procs() 1
#endif

namespace tug {

/**
 * @brief The class forms the interface for performing the diffusion simulations
 * and contains all the methods for controlling the desired parameters, such as
 * time step, number of simulations, etc.
 *
 * @tparam T the type of the internal data structures for grid, boundary
 * condition and timestep
 */
template <class T>
class Simulation {
public:
  /**
   * @brief Set up a simulation environment. The timestep and number of
   * iterations must be set. For the BTCS approach, the Thomas algorithm is used
   * as the default linear equation solver as this is faster for tridiagonal
   *        matrices. CSV output, console output and time measure are off by
   * default. Also, the number of cores is set to the maximum number of cores -1
   * by default.
   *
   * @param grid Valid grid object
   * @param bc Valid boundary condition object
   */
  Simulation(Grid<T> &_grid, Boundary<T> &_bc) : grid(_grid), bc(_bc){};

  /**
   * @brief Setting the time step for each iteration step. Time step must be
   *        greater than zero. Setting the timestep is required.
   *
   * @param timestep Valid timestep greater than zero.
   */
  void setTimestep(T timestep) {
    if (timestep <= 0) {
      throw_invalid_argument("Timestep has to be greater than zero.");
    }

    {
      this->timestep = timestep;
    }
  }

  /**
   * @brief Currently set time step is returned.
   *
   * @return double timestep
   */
  T getTimestep() const { return this->timestep; }

  /**
   * @brief Set the desired iterations to be calculated. A value greater
   *        than zero must be specified here. Setting iterations is required.
   *
   * @param iterations Number of iterations to be simulated.
   */
  void setIterations(int iterations) {
    if (iterations < 0) {
      throw std::invalid_argument(
          "Number of iterations must be zero or greater.");
    }
    this->iterations = iterations;
  }

  /**
   * @brief Return the currently set iterations to be calculated.
   *
   * @return int Number of iterations.
   */
  int getIterations() const { return this->iterations; }

  /**
   * @brief Method starts the simulation process with the previously set
   *        parameters.
   */
  void run() {
    if (this->timestep == -1) {
      throw_invalid_argument("Timestep is not set!");
    }
    if (this->iterations < 1) {
      throw_invalid_argument("Number of iterations are not properly set!");
    }

    BTCS_Thomas(this->grid, this->bc, this->timestep, this->iterations);
  }

private:
  T timestep{-1};
  int iterations{-1};
  int innerIterations{1};

  Grid<T> &grid;
  Boundary<T> &bc;
};
} // namespace tug
#endif // SIMULATION_H_
