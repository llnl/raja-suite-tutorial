#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <caliper/cali.h>

#include "RAJA/RAJA.hpp"
#include "RAJA/pattern/synchronize.hpp"
#include "caliper-plugin.cpp"
#include "umpire/Umpire.hpp"

// Basic Caliper profiling:
//   CALI_CONFIG=runtime-report ./bin/profile_raja 1024
//
// H100 recommendation:
//   GLOBAL 32x8 is the preferred starting policy for this naive
//   matrix multiply. It uses 256 threads per block and maps adjacent
//   threads to adjacent output columns.
//
// The actual winner depends on matrix size and compiler behavior, so
// this program benchmarks all variants and reports the results.
//
// Important:
//   This is a naive global-memory matrix multiply. A shared-memory tiled
//   kernel or cuBLAS GEMM will provide substantially better H100 performance.

namespace device {
constexpr bool async = false;
constexpr int threads_per_block = 256;

#if defined(RAJA_ENABLE_CUDA)
using forall_pol = RAJA::cuda_exec<threads_per_block, async>;
using sync_pol = RAJA::cuda_synchronize;

using launch_no_lb_pol = RAJA::LaunchPolicy<RAJA::cuda_launch_t<async>>;

template <int num_threads>
using launch_lb_pol =
    RAJA::LaunchPolicy<RAJA::cuda_launch_t<async, num_threads>>;

using block_x_direct = RAJA::cuda_block_x_direct;
using thread_x_direct = RAJA::cuda_thread_x_direct;
using block_x_loop = RAJA::cuda_block_x_loop;
using thread_x_loop = RAJA::cuda_thread_x_loop;

template <int block_x>
using global_size_x_direct = RAJA::cuda_global_size_x_direct<block_x>;
template <int block_y>
using global_size_y_direct = RAJA::cuda_global_size_y_direct<block_y>;
#elif defined(RAJA_ENABLE_HIP)
using forall_pol = RAJA::hip_exec<threads_per_block, async>;
using sync_pol = RAJA::hip_synchronize;

using launch_no_lb_pol = RAJA::LaunchPolicy<RAJA::hip_launch_t<async>>;

template <int num_threads>
using launch_lb_pol =
    RAJA::LaunchPolicy<RAJA::hip_launch_t<async, num_threads>>;

using block_x_direct = RAJA::hip_block_x_direct;
using thread_x_direct = RAJA::hip_thread_x_direct;
using block_x_loop = RAJA::hip_block_x_loop;
using thread_x_loop = RAJA::hip_thread_x_loop;

template <int block_x>
using global_size_x_direct = RAJA::hip_global_size_x_direct<block_x>;
template <int block_y>
using global_size_y_direct = RAJA::hip_global_size_y_direct<block_y>;
#else
#error "profile_raja requires RAJA_ENABLE_CUDA or RAJA_ENABLE_HIP."
#endif
} // namespace device

enum class Stage { baseline, sweep, single, list };

enum class Group { all, direct, loop, global };

struct Options {
  int n = 0;
  Stage stage = Stage::baseline;
  Group group = Group::all;
  std::string variant_region;
  int warmup_runs = 5;
  int profiled_runs = 20;
  int baseline_profiled_runs = 2;
  bool validate = true;
};

static void print_usage(const char *argv0) {
  std::cout << "Usage:\n"
            << "  " << argv0 << " N [options]\n\n"
            << "Options:\n"
            << "  --stage <baseline|sweep|single|list>   (default: baseline)\n"
            << "  --group <all|direct|loop|global>       (default: all)\n"
            << "  --variant <region_name>               (required for --stage "
               "single)\n"
            << "  --warmup <int>                        (default: 5)\n"
            << "  --runs <int>                          (default: 20)\n"
            << "  --baseline-runs <int>                 (default: 2)\n"
            << "  --validate <0|1>                      (default: 1)\n";
}

static std::optional<int> parse_int(std::string_view s) {
  if (s.empty()) {
    return std::nullopt;
  }

  char *end = nullptr;
  const long v = std::strtol(std::string(s).c_str(), &end, 10);
  if (!end || *end != '\0') {
    return std::nullopt;
  }
  if (v < std::numeric_limits<int>::min() ||
      v > std::numeric_limits<int>::max()) {
    return std::nullopt;
  }
  return static_cast<int>(v);
}

static Stage parse_stage(std::string_view s) {
  if (s == "baseline")
    return Stage::baseline;
  if (s == "sweep")
    return Stage::sweep;
  if (s == "single")
    return Stage::single;
  if (s == "list")
    return Stage::list;
  throw std::runtime_error("Invalid --stage value: " + std::string(s));
}

static Group parse_group(std::string_view s) {
  if (s == "all")
    return Group::all;
  if (s == "direct")
    return Group::direct;
  if (s == "loop")
    return Group::loop;
  if (s == "global")
    return Group::global;
  throw std::runtime_error("Invalid --group value: " + std::string(s));
}

static Options parse_options(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    throw std::runtime_error("Missing required N");
  }

  Options opt{};
  {
    auto n_opt = parse_int(argv[1]);
    if (!n_opt || *n_opt <= 0) {
      throw std::runtime_error("Matrix size N must be a positive integer");
    }
    opt.n = *n_opt;
  }

  for (int i = 2; i < argc; ++i) {
    const std::string_view arg(argv[i]);

    auto need_value = [&](std::string_view flag) -> std::string_view {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + std::string(flag));
      }
      return std::string_view(argv[++i]);
    };

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--stage") {
      opt.stage = parse_stage(need_value(arg));
    } else if (arg == "--group") {
      opt.group = parse_group(need_value(arg));
    } else if (arg == "--variant") {
      opt.variant_region = std::string(need_value(arg));
    } else if (arg == "--warmup") {
      auto v = parse_int(need_value(arg));
      if (!v || *v < 0)
        throw std::runtime_error("Invalid --warmup value");
      opt.warmup_runs = *v;
    } else if (arg == "--runs") {
      auto v = parse_int(need_value(arg));
      if (!v || *v <= 0)
        throw std::runtime_error("Invalid --runs value");
      opt.profiled_runs = *v;
    } else if (arg == "--baseline-runs") {
      auto v = parse_int(need_value(arg));
      if (!v || *v < 0)
        throw std::runtime_error("Invalid --baseline-runs value");
      opt.baseline_profiled_runs = *v;
    } else if (arg == "--validate") {
      auto v = parse_int(need_value(arg));
      if (!v || (*v != 0 && *v != 1))
        throw std::runtime_error("Invalid --validate value");
      opt.validate = (*v == 1);
    } else {
      throw std::runtime_error("Unknown argument: " + std::string(arg));
    }
  }

  if (opt.stage == Stage::single && opt.variant_region.empty()) {
    throw std::runtime_error("--stage single requires --variant <region_name>");
  }

  return opt;
}

void init(double *A, double *B, double *C, int m, int n, bool profile) {
  const RAJA::Index_type count = static_cast<RAJA::Index_type>(m) * n;

  RAJA::forall<device::forall_pol>(RAJA::RangeSegment(0, count),
                                   RAJA::Name(profile ? "init" : ""),
                                   [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
                                     A[i] = 1.0;
                                     B[i] = 1.0;
                                     C[i] = 0.0;
                                   });
}

void matrix_add(const double *A, const double *B, double *C, int m, int n,
                bool profile) {
  const RAJA::Index_type count = static_cast<RAJA::Index_type>(m) * n;

  RAJA::forall<device::forall_pol>(
      RAJA::RangeSegment(0, count), RAJA::Name(profile ? "matrix_add" : ""),
      [=] RAJA_HOST_DEVICE(RAJA::Index_type i) { C[i] = A[i] + B[i]; });
}

void matrix_scalar_mult(const double *A, double *B, double scalar, int m, int n,
                        bool profile) {
  const RAJA::Index_type count = static_cast<RAJA::Index_type>(m) * n;

  RAJA::forall<device::forall_pol>(
      RAJA::RangeSegment(0, count),
      RAJA::Name(profile ? "matrix_scalar_mult" : ""),
      [=] RAJA_HOST_DEVICE(RAJA::Index_type i) { B[i] = scalar * A[i]; });
}

template <typename LaunchPolicy, typename Loop1Policy, typename Loop0Policy>
void matrix_multiply_variant(const double *A, const double *B, double *C, int m,
                             int n, int p, const char *name,
                             const RAJA::LaunchParams &params) {
  // A: m x n
  // B: n x p
  // C: m x p

  auto v_A = RAJA::make_permuted_view<RAJA::layout_right>(A, m, n);

  auto v_B = RAJA::make_permuted_view<RAJA::layout_right>(B, n, p);

  auto v_C = RAJA::make_permuted_view<RAJA::layout_right>(C, m, p);

  RAJA::launch<LaunchPolicy>(
      params, RAJA::Name(name), [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::loop<Loop1Policy>(ctx, RAJA::RangeSegment(0, m), [=](int i) {
          RAJA::loop<Loop0Policy>(ctx, RAJA::RangeSegment(0, p), [=](int j) {
            double dot = 0.0;

            for (int k = 0; k < n; ++k) {
              dot += v_A(i, k) * v_B(k, j);
            }

            v_C(i, j) = dot;
          });
        });
      });
}

template <typename LaunchPolicy>
void matrix_multiply_direct(const double *A, const double *B, double *C, int m,
                            int n, int p, const char *name) {
  // DIRECT uses one block per row and one thread per column.
  // It is appropriate only when p <= the device block-thread limit.

  if (p > 1024) {
    throw std::runtime_error("DIRECT requires p <= 1024");
  }

  using loop1_pol = RAJA::LoopPolicy<device::block_x_direct>;

  using loop0_pol = RAJA::LoopPolicy<device::thread_x_direct>;

  RAJA::LaunchParams params{RAJA::Teams(m), RAJA::Threads(p)};

  matrix_multiply_variant<LaunchPolicy, loop1_pol, loop0_pol>(A, B, C, m, n, p,
                                                              name, params);
}

template <typename LaunchPolicy>
void matrix_multiply_loop(const double *A, const double *B, double *C, int m,
                          int n, int p, const char *name) {
  // LOOP is a general baseline using 256 threads per block.

  using loop1_pol = RAJA::LoopPolicy<device::block_x_loop>;

  using loop0_pol = RAJA::LoopPolicy<device::thread_x_loop>;

  RAJA::LaunchParams params{RAJA::Teams(m), RAJA::Threads(256)};

  matrix_multiply_variant<LaunchPolicy, loop1_pol, loop0_pol>(A, B, C, m, n, p,
                                                              name, params);
}

template <typename LaunchPolicy>
void matrix_multiply_global_16x16(const double *A, const double *B, double *C,
                                  int m, int n, int p, const char *name) {
  // 16 x 16 = 256 threads per block.
  // This is a useful comparison policy.

  constexpr int block_x = 16;
  constexpr int block_y = 16;

  using loop1_pol = RAJA::LoopPolicy<device::global_size_y_direct<block_y>>;

  using loop0_pol = RAJA::LoopPolicy<device::global_size_x_direct<block_x>>;

  const int teams_x = (p + block_x - 1) / block_x;

  const int teams_y = (m + block_y - 1) / block_y;

  RAJA::LaunchParams params{RAJA::Teams(teams_x, teams_y),
                            RAJA::Threads(block_x, block_y)};

  matrix_multiply_variant<LaunchPolicy, loop1_pol, loop0_pol>(A, B, C, m, n, p,
                                                              name, params);
}

template <typename LaunchPolicy>
void matrix_multiply_global_32x8(const double *A, const double *B, double *C,
                                 int m, int n, int p, const char *name) {
  // Recommended H100 starting point.
  //
  // 32 x 8 = 256 threads per block.
  // The X dimension is warp-aligned and maps adjacent threads
  // to adjacent output columns.

  constexpr int block_x = 32;
  constexpr int block_y = 8;

  using loop1_pol = RAJA::LoopPolicy<device::global_size_y_direct<block_y>>;

  using loop0_pol = RAJA::LoopPolicy<device::global_size_x_direct<block_x>>;

  const int teams_x = (p + block_x - 1) / block_x;

  const int teams_y = (m + block_y - 1) / block_y;

  RAJA::LaunchParams params{RAJA::Teams(teams_x, teams_y),
                            RAJA::Threads(block_x, block_y)};

  matrix_multiply_variant<LaunchPolicy, loop1_pol, loop0_pol>(A, B, C, m, n, p,
                                                              name, params);
}

template <typename LaunchPolicy>
void matrix_multiply_global_32x16(const double *A, const double *B, double *C,
                                  int m, int n, int p, const char *name) {
  // 32 x 16 = 512 threads per block.
  // This may be slower than 32x8 because larger blocks can reduce
  // occupancy, but it is included for benchmarking.

  constexpr int block_x = 32;
  constexpr int block_y = 16;

  using loop1_pol = RAJA::LoopPolicy<device::global_size_y_direct<block_y>>;

  using loop0_pol = RAJA::LoopPolicy<device::global_size_x_direct<block_x>>;

  const int teams_x = (p + block_x - 1) / block_x;

  const int teams_y = (m + block_y - 1) / block_y;

  RAJA::LaunchParams params{RAJA::Teams(teams_x, teams_y),
                            RAJA::Threads(block_x, block_y)};

  matrix_multiply_variant<LaunchPolicy, loop1_pol, loop0_pol>(A, B, C, m, n, p,
                                                              name, params);
}

bool check_matrix_multiply(const double *C, int m, int p, int n) {
  auto v_C = RAJA::make_permuted_view<RAJA::layout_right>(C, m, p);

  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < p; ++c) {
      const double expected = static_cast<double>(n);

      const double actual = v_C(r, c);

      if (std::abs(actual - expected) > 1.0e-10) {
        std::cerr << "Validation failure at (" << r << ", " << c
                  << "): expected " << expected << ", got " << actual << '\n';

        return false;
      }
    }
  }

  return true;
}

struct Variant {
  const char *label;
  const char *region;
  Group group;
  enum class Kind {
    direct,
    loop,
    global_16x16,
    global_32x8,
    global_32x16
  } kind;

  enum class Bounds { none, launch_bounds } bounds;
};

static bool group_allows(Group filter, Group variant_group) {
  return filter == Group::all || filter == variant_group;
}

static const char *region_name(const Variant &v, bool profile) {
  return profile ? v.region : "";
}

static void run_matmul_variant(const Variant &v, bool profile, const double *A,
                               const double *B, double *C, int m, int n,
                               int p) {
  const char *name = region_name(v, profile);

  switch (v.kind) {
  case Variant::Kind::direct:
    if (v.bounds == Variant::Bounds::none) {
      matrix_multiply_direct<device::launch_no_lb_pol>(A, B, C, m, n, p, name);
    } else {
      matrix_multiply_direct<device::launch_lb_pol<1024>>(A, B, C, m, n, p,
                                                          name);
    }
    break;

  case Variant::Kind::loop:
    if (v.bounds == Variant::Bounds::none) {
      matrix_multiply_loop<device::launch_no_lb_pol>(A, B, C, m, n, p, name);
    } else {
      matrix_multiply_loop<device::launch_lb_pol<256>>(A, B, C, m, n, p, name);
    }
    break;

  case Variant::Kind::global_16x16:
    if (v.bounds == Variant::Bounds::none) {
      matrix_multiply_global_16x16<device::launch_no_lb_pol>(A, B, C, m, n, p,
                                                             name);
    } else {
      matrix_multiply_global_16x16<device::launch_lb_pol<256>>(A, B, C, m, n, p,
                                                               name);
    }
    break;

  case Variant::Kind::global_32x8:
    if (v.bounds == Variant::Bounds::none) {
      matrix_multiply_global_32x8<device::launch_no_lb_pol>(A, B, C, m, n, p,
                                                            name);
    } else {
      matrix_multiply_global_32x8<device::launch_lb_pol<256>>(A, B, C, m, n, p,
                                                              name);
    }
    break;

  case Variant::Kind::global_32x16:
    if (v.bounds == Variant::Bounds::none) {
      matrix_multiply_global_32x16<device::launch_no_lb_pol>(A, B, C, m, n, p,
                                                             name);
    } else {
      matrix_multiply_global_32x16<device::launch_lb_pol<512>>(A, B, C, m, n, p,
                                                               name);
    }
    break;
  }
}

static std::vector<Variant> make_variants(int n) {
  std::vector<Variant> variants;
  variants.reserve(10);

  if (n <= 1024) {
    variants.push_back(Variant{"DIRECT (no launch bounds)",
                               "matrix_multiply_direct.no_lb", Group::direct,
                               Variant::Kind::direct, Variant::Bounds::none});

    variants.push_back(Variant{
        "DIRECT (launch bounds)", "matrix_multiply_direct.lb", Group::direct,
        Variant::Kind::direct, Variant::Bounds::launch_bounds});
  }

  variants.push_back(Variant{"LOOP (no launch bounds)",
                             "matrix_multiply_loop.no_lb", Group::loop,
                             Variant::Kind::loop, Variant::Bounds::none});

  variants.push_back(Variant{"LOOP (launch bounds)", "matrix_multiply_loop.lb",
                             Group::loop, Variant::Kind::loop,
                             Variant::Bounds::launch_bounds});

  variants.push_back(Variant{
      "GLOBAL 16x16 (no launch bounds)", "matrix_multiply_global_16x16.no_lb",
      Group::global, Variant::Kind::global_16x16, Variant::Bounds::none});

  variants.push_back(Variant{"GLOBAL 16x16 (launch bounds)",
                             "matrix_multiply_global_16x16.lb", Group::global,
                             Variant::Kind::global_16x16,
                             Variant::Bounds::launch_bounds});

  variants.push_back(Variant{
      "GLOBAL 32x8 (no launch bounds)", "matrix_multiply_global_32x8.no_lb",
      Group::global, Variant::Kind::global_32x8, Variant::Bounds::none});

  variants.push_back(Variant{"GLOBAL 32x8 (launch bounds)",
                             "matrix_multiply_global_32x8.lb", Group::global,
                             Variant::Kind::global_32x8,
                             Variant::Bounds::launch_bounds});

  variants.push_back(Variant{
      "GLOBAL 32x16 (no launch bounds)", "matrix_multiply_global_32x16.no_lb",
      Group::global, Variant::Kind::global_32x16, Variant::Bounds::none});

  variants.push_back(Variant{"GLOBAL 32x16 (launch bounds)",
                             "matrix_multiply_global_32x16.lb", Group::global,
                             Variant::Kind::global_32x16,
                             Variant::Bounds::launch_bounds});

  return variants;
}

static std::unordered_map<std::string, double>
parse_caliper_json_table(const std::string &json, std::string_view key_field,
                         std::string_view value_field) {
  std::unordered_map<std::string, double> out;

  std::size_t pos = 0;
  while (true) {
    const std::size_t begin = json.find('{', pos);
    if (begin == std::string::npos) {
      break;
    }
    const std::size_t end = json.find('}', begin);
    if (end == std::string::npos) {
      break;
    }

    const std::string_view obj(json.data() + begin, end - begin + 1);

    auto find_string_field =
        [&](std::string_view field) -> std::optional<std::string> {
      const std::string needle = "\"" + std::string(field) + "\"";
      const std::size_t k = obj.find(needle);
      if (k == std::string::npos)
        return std::nullopt;
      const std::size_t colon = obj.find(':', k + needle.size());
      if (colon == std::string::npos)
        return std::nullopt;
      const std::size_t q1 = obj.find('"', colon + 1);
      if (q1 == std::string::npos)
        return std::nullopt;
      const std::size_t q2 = obj.find('"', q1 + 1);
      if (q2 == std::string::npos)
        return std::nullopt;
      return std::string(obj.substr(q1 + 1, q2 - q1 - 1));
    };

    auto find_number_field =
        [&](std::string_view field) -> std::optional<double> {
      const std::string needle = "\"" + std::string(field) + "\"";
      const std::size_t k = obj.find(needle);
      if (k == std::string::npos)
        return std::nullopt;
      const std::size_t colon = obj.find(':', k + needle.size());
      if (colon == std::string::npos)
        return std::nullopt;
      const std::size_t num_begin =
          obj.find_first_of("+-0123456789.", colon + 1);
      if (num_begin == std::string::npos)
        return std::nullopt;
      const char *start = obj.data() + num_begin;
      char *num_end = nullptr;
      const double v = std::strtod(start, &num_end);
      if (num_end == start)
        return std::nullopt;
      return v;
    };

    auto key = find_string_field(key_field);
    auto value = find_number_field(value_field);

    if (key && value) {
      out.emplace(std::move(*key), *value);
    }

    pos = end + 1;
  }

  return out;
}

static void print_winners_from_caliper(cali_id_t channel_id,
                                       const std::vector<Variant> &variants,
                                       int profiled_runs) {
  // Query total seconds per region name ("path").
  const char *query = "let t=scale(sum#time.duration.ns,1e-9),l=leaf() where l "
                      "group by path select sum(t) as \"time_s\" format json";

  std::ostringstream os;
  cali::write_report_for_query(channel_id, query, CALI_FLUSH_CLEAR_BUFFERS, os);

  const auto table = parse_caliper_json_table(os.str(), "path", "time_s");

  auto best_of =
      [&](Group gfilter) -> std::optional<std::pair<std::string, double>> {
    std::optional<std::pair<std::string, double>> best;
    for (const Variant &v : variants) {
      if (gfilter != Group::all && v.group != gfilter) {
        continue;
      }

      auto it = table.find(v.region);
      if (it == table.end()) {
        continue;
      }

      const double avg = it->second / static_cast<double>(profiled_runs);
      if (!best || avg < best->second) {
        best = std::make_pair(std::string(v.region), avg);
      }
    }
    return best;
  };

  const auto best_overall = best_of(Group::all);
  const auto best_global = best_of(Group::global);

  std::cout << "\nWinners (from Caliper, avg seconds/run)\n"
            << "--------------------------------------\n";

  if (best_overall) {
    std::cout << "Fastest overall: " << best_overall->first << "  ("
              << std::fixed << std::setprecision(6) << best_overall->second
              << " s/run)\n";
  } else {
    std::cout << "Fastest overall: (no Caliper data found)\n";
  }

  if (best_global) {
    std::cout << "Fastest GLOBAL:  " << best_global->first << "  ("
              << std::fixed << std::setprecision(6) << best_global->second
              << " s/run)\n";
  } else {
    std::cout << "Fastest GLOBAL:  (no Caliper data found)\n";
  }
}

int main(int argc, char *argv[]) {
  try {
    const Options opt = parse_options(argc, argv);
    const int n = opt.n;

    // Create a dedicated Caliper channel so we can auto-pick a winner using a
    // CalQL query without introducing non-Caliper timing code paths.
    // Use `trace` here (not `aggregate`) so the in-process CalQL query can
    // compute per-region stats without requiring a pre-aggregation key.
    cali::config_map_t cfg{
        {"CALI_SERVICES_ENABLE", "event,timer,trace"},
        {"CALI_TIMER_UNIT", "sec"},
    };
    const cali_id_t demo_channel = cali::create_channel("policy-demo", 0, cfg);

    const auto all_variants = make_variants(n);

    if (opt.stage == Stage::list) {
      std::cout << "Available variants\n"
                << "------------------\n";
      for (const Variant &v : all_variants) {
        if (!group_allows(opt.group, v.group)) {
          continue;
        }
        std::cout << std::left << std::setw(34) << v.region << "  " << v.label
                  << '\n';
      }
      return 0;
    }

    std::cout << "Using matrix size " << n << " x " << n << '\n'
              << "Warm-up runs: " << opt.warmup_runs << '\n'
              << "Profiled runs: " << opt.profiled_runs << "\n\n";

    auto &resource_manager = umpire::ResourceManager::getInstance();

    auto host_allocator = resource_manager.getAllocator("HOST");

    auto device_allocator = resource_manager.getAllocator("DEVICE");

    const std::size_t elements = static_cast<std::size_t>(n) * n;

    const std::size_t bytes = elements * sizeof(double);

    double *h_C = static_cast<double *>(host_allocator.allocate(bytes));

    double *d_A = static_cast<double *>(device_allocator.allocate(bytes));

    double *d_B = static_cast<double *>(device_allocator.allocate(bytes));

    double *d_C = static_cast<double *>(device_allocator.allocate(bytes));

    std::cout << "Profiling regions\n"
              << "-----------------\n"
              << "- init\n"
              << "- matrix_add\n"
              << "- matrix_scalar_mult\n";

    for (const Variant &variant : all_variants) {
      if (!group_allows(opt.group, variant.group)) {
        continue;
      }
      std::cout << "- " << variant.label << '\n';
    }

    std::cout << '\n';

    // Profile a couple of non-matmul kernels so matrix multiply shows up
    // clearly as the hotspot in Caliper reports.
    init(d_A, d_B, d_C, n, n, true);

    for (int i = 0; i < opt.baseline_profiled_runs; ++i) {
      matrix_add(d_A, d_B, d_C, n, n, true);
      matrix_scalar_mult(d_C, d_C, 0.5, n, n, true);
    }

    auto find_variant_by_region =
        [&](std::string_view region) -> const Variant * {
      for (const Variant &v : all_variants) {
        if (region == v.region) {
          return &v;
        }
      }
      return nullptr;
    };

    if (opt.stage == Stage::baseline) {
      const Variant *base =
          find_variant_by_region("matrix_multiply_loop.no_lb");
      if (!base) {
        throw std::runtime_error(
            "Baseline variant missing: matrix_multiply_loop.no_lb");
      }

      init(d_A, d_B, d_C, n, n, false);

      for (int i = 0; i < opt.warmup_runs; ++i) {
        run_matmul_variant(*base, false, d_A, d_B, d_C, n, n, n);
      }

      for (int i = 0; i < opt.profiled_runs; ++i) {
        run_matmul_variant(*base, true, d_A, d_B, d_C, n, n, n);
      }

      RAJA::synchronize<device::sync_pol>();

      resource_manager.copy(h_C, d_C, bytes);
      RAJA::synchronize<device::sync_pol>();

      if (opt.validate && !check_matrix_multiply(h_C, n, n, n)) {
        throw std::runtime_error("Validation failed for baseline");
      }

      std::cout << "\nBaseline complete.\n"
                << "Next: run `--stage sweep` to explore policies live.\n";
    } else if (opt.stage == Stage::single) {
      const Variant *v = find_variant_by_region(opt.variant_region);
      if (!v) {
        throw std::runtime_error("Unknown --variant region: " +
                                 opt.variant_region);
      }
      if (!group_allows(opt.group, v->group)) {
        throw std::runtime_error("Selected --variant is excluded by --group");
      }

      init(d_A, d_B, d_C, n, n, false);

      for (int i = 0; i < opt.warmup_runs; ++i) {
        run_matmul_variant(*v, false, d_A, d_B, d_C, n, n, n);
      }

      for (int i = 0; i < opt.profiled_runs; ++i) {
        run_matmul_variant(*v, true, d_A, d_B, d_C, n, n, n);
      }

      RAJA::synchronize<device::sync_pol>();

      resource_manager.copy(h_C, d_C, bytes);
      RAJA::synchronize<device::sync_pol>();

      if (opt.validate && !check_matrix_multiply(h_C, n, n, n)) {
        throw std::runtime_error("Validation failed for " +
                                 std::string(v->label));
      }
    } else if (opt.stage == Stage::sweep) {
      std::cout << "\nMatrix multiply tuning sweep\n"
                << "----------------------------\n";

      std::vector<Variant> variants;
      for (const Variant &v : all_variants) {
        if (group_allows(opt.group, v.group)) {
          variants.push_back(v);
        }
      }

      for (const Variant &variant : variants) {
        // Reinitialize before each variant so every test starts with identical
        // input and does not depend on a prior result.

        init(d_A, d_B, d_C, n, n, false);

        for (int i = 0; i < opt.warmup_runs; ++i) {
          run_matmul_variant(variant, false, d_A, d_B, d_C, n, n, n);
        }

        for (int i = 0; i < opt.profiled_runs; ++i) {
          run_matmul_variant(variant, true, d_A, d_B, d_C, n, n, n);
        }

        RAJA::synchronize<device::sync_pol>();

        if (opt.validate) {
          resource_manager.copy(h_C, d_C, bytes);
          RAJA::synchronize<device::sync_pol>();

          if (!check_matrix_multiply(h_C, n, n, n)) {
            throw std::runtime_error(std::string("Validation failed for ") +
                                     variant.label);
          }
        }
      }

      print_winners_from_caliper(demo_channel, variants, opt.profiled_runs);
    }

    if (opt.stage == Stage::sweep) {
      std::cout << "\nSweep complete.\n"
                << "For this naive matrix multiply, GLOBAL policies are good "
                   "starting points.\n"
                << "Use Caliper (e.g., `CALI_CONFIG=runtime-report`) for "
                   "detailed timings.\n";
    } else {
      std::cout
          << "\nRun complete.\n"
          << "Use Caliper (e.g., `CALI_CONFIG=runtime-report`) for timings.\n";
    }

    host_allocator.deallocate(h_C);

    device_allocator.deallocate(d_A);
    device_allocator.deallocate(d_B);
    device_allocator.deallocate(d_C);

    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';

    return 1;
  }
}
