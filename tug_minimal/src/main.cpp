#include "bench_defs.hpp"
#include "run.hpp"

#include <CLI/CLI.hpp>
#include <iostream>
#include <string>
#include <tug/Simulation.hpp>

using TugType = double;

int main(int argc, char **argv) {
  const bench_input &benchmark = barite_large_input;

  CLI::App app{"Run the diffusion benchmark"};
  int iterations = 0;
  std::string output_file;

  app.add_option("iterations", iterations,
                 "Number of iterations to execute for the benchmark")
      ->required();
  app.add_option("-o,--output", output_file, "Output CSV file");

  CLI11_PARSE(app, argc, argv);

  if (output_file.empty()) {
    output_file = benchmark.benchmark_name + "_out.csv";
  }

  std::cout << benchmark.benchmark_name << " started...\n";
  run_bench(benchmark, output_file, iterations);

  return EXIT_SUCCESS;
}
