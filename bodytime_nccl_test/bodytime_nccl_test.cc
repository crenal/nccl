#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define CUDACHECK(cmd) do { \
  cudaError_t e = (cmd); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA failure %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

#define NCCLCHECK(cmd) do { \
  ncclResult_t r = (cmd); \
  if (r != ncclSuccess) { \
    fprintf(stderr, "NCCL failure %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
    MPI_Abort(MPI_COMM_WORLD, 2); \
  } \
} while (0)

struct Options {
  size_t recvBytes = 4ULL * 1024ULL * 1024ULL;
  int warmup = 100;
  int iters = 1000;
  int device = -1;
  double ticksPerUs = 1000.0;
  bool keepProfile = false;
  std::string profileDir;
};

static void usage(const char* argv0) {
  fprintf(stderr,
          "Usage: %s [--recv-bytes <bytes>] [--bytes <bytes>] [--warmup <n>] [--iters <n>] "
          "[--device <id>] [--ticks-per-us <x>] [--profile-dir <dir>] [--keep-profile]\n",
          argv0);
}

static size_t parseSize(const char* s) {
  char* end = nullptr;
  double value = strtod(s, &end);
  if (end == s || value < 0) return 0;
  size_t scale = 1;
  if (*end != '\0') {
    if ((end[0] == 'K' || end[0] == 'k') && end[1] == '\0') scale = 1024ULL;
    else if ((end[0] == 'M' || end[0] == 'm') && end[1] == '\0') scale = 1024ULL * 1024ULL;
    else if ((end[0] == 'G' || end[0] == 'g') && end[1] == '\0') scale = 1024ULL * 1024ULL * 1024ULL;
    else return 0;
  }
  return static_cast<size_t>(value * scale);
}

static Options parseArgs(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    auto needValue = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        fprintf(stderr, "%s requires a value\n", name);
        usage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 3);
      }
      return argv[++i];
    };
    if (strcmp(argv[i], "--recv-bytes") == 0 || strcmp(argv[i], "--bytes") == 0) {
      opt.recvBytes = parseSize(needValue(argv[i]));
    } else if (strcmp(argv[i], "--warmup") == 0) {
      opt.warmup = atoi(needValue(argv[i]));
    } else if (strcmp(argv[i], "--iters") == 0) {
      opt.iters = atoi(needValue(argv[i]));
    } else if (strcmp(argv[i], "--device") == 0) {
      opt.device = atoi(needValue(argv[i]));
    } else if (strcmp(argv[i], "--ticks-per-us") == 0) {
      opt.ticksPerUs = atof(needValue(argv[i]));
    } else if (strcmp(argv[i], "--profile-dir") == 0) {
      opt.profileDir = needValue(argv[i]);
    } else if (strcmp(argv[i], "--keep-profile") == 0) {
      opt.keepProfile = true;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      MPI_Finalize();
      exit(0);
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      usage(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
  }
  if (opt.recvBytes == 0 || opt.warmup < 0 || opt.iters <= 0 || opt.ticksPerUs <= 0) {
    fprintf(stderr, "Invalid arguments\n");
    MPI_Abort(MPI_COMM_WORLD, 3);
  }
  return opt;
}

static void cleanupProfileDir(const std::string& dir) {
  DIR* dp = opendir(dir.c_str());
  if (dp == nullptr) return;
  while (dirent* de = readdir(dp)) {
    std::string name = de->d_name;
    if (name == "." || name == "..") continue;
    unlink((dir + "/" + name).c_str());
  }
  closedir(dp);
  rmdir(dir.c_str());
}

static int envInt(const char* name, int fallback) {
  const char* s = getenv(name);
  return s ? atoi(s) : fallback;
}

static void mkdirOne(const std::string& path) {
  if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
    fprintf(stderr, "Could not mkdir %s: %s\n", path.c_str(), strerror(errno));
    MPI_Abort(MPI_COMM_WORLD, 4);
  }
}

static std::string readFile(const std::string& path) {
  std::ifstream f(path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static bool extractU64After(const std::string& text, const std::string& key, size_t pos, uint64_t* out) {
  size_t k = text.find(key, pos);
  if (k == std::string::npos) return false;
  k = text.find(':', k);
  if (k == std::string::npos) return false;
  k++;
  while (k < text.size() && (text[k] == ' ' || text[k] == '"')) k++;
  size_t e = k;
  while (e < text.size() && text[e] >= '0' && text[e] <= '9') e++;
  if (e == k) return false;
  *out = strtoull(text.substr(k, e - k).c_str(), nullptr, 10);
  return true;
}

static int dumpIdFromName(const std::string& name) {
  size_t p = name.find(".dump");
  if (p == std::string::npos) return -1;
  p += 5;
  size_t e = name.find(".json", p);
  if (e == std::string::npos) return -1;
  return atoi(name.substr(p, e - p).c_str());
}

static std::vector<double> parseBodyTimesUs(const std::string& dir, int warmup, int iters, double ticksPerUs) {
  struct Entry {
    int dump;
    double us;
  };
  std::vector<Entry> entries;

  DIR* dp = opendir(dir.c_str());
  if (dp == nullptr) return {};
  while (dirent* de = readdir(dp)) {
    std::string name = de->d_name;
    if (name.find(".json") == std::string::npos) continue;
    int dump = dumpIdFromName(name);
    if (dump < 0) continue;

    std::string text = readFile(dir + "/" + name);
    size_t pos = 0;
    uint64_t maxCycles = 0;
    bool found = false;
    while ((pos = text.find("\"event_type\":\"body_time\"", pos)) != std::string::npos) {
      uint64_t cycles = 0;
      if (extractU64After(text, "\"elapsed_cycles\"", pos, &cycles)) {
        maxCycles = std::max(maxCycles, cycles);
        found = true;
      }
      pos += 24;
    }
    if (found) entries.push_back({dump, static_cast<double>(maxCycles) / ticksPerUs});
  }
  closedir(dp);

  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
    return a.dump < b.dump;
  });

  std::vector<double> values;
  for (const Entry& e : entries) {
    if (e.dump < warmup) continue;
    if ((int)values.size() >= iters) break;
    values.push_back(e.us);
  }
  return values;
}

static double percentile(const std::vector<double>& v, double q) {
  if (v.empty()) return 0.0;
  size_t idx = static_cast<size_t>((v.size() - 1) * q + 0.5);
  if (idx >= v.size()) idx = v.size() - 1;
  return v[idx];
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  Options opt = parseArgs(argc, argv);
  int ndev = 0;
  CUDACHECK(cudaGetDeviceCount(&ndev));
  int localRank = envInt("OMPI_COMM_WORLD_LOCAL_RANK", envInt("MV2_COMM_WORLD_LOCAL_RANK", rank));
  int device = opt.device >= 0 ? opt.device : (localRank % std::max(1, ndev));
  CUDACHECK(cudaSetDevice(device));

  if (opt.recvBytes % nranks != 0) {
    if (rank == 0) fprintf(stderr, "recvBytes must be divisible by nranks\n");
    MPI_Abort(MPI_COMM_WORLD, 5);
  }
  size_t sendBytes = opt.recvBytes / nranks;
  size_t recvBytes = sendBytes * nranks;

  char host[256];
  gethostname(host, sizeof(host));
  host[sizeof(host) - 1] = '\0';
  if (opt.profileDir.empty()) {
    opt.profileDir = "/tmp/bodytime_nccl_test_" + std::to_string(getpid()) + "_rank" + std::to_string(rank);
  }
  mkdirOne(opt.profileDir);
  std::string prefix = opt.profileDir + "/ag_body";
  setenv("NCCL_SYM_AG_GIN_PROFILE_JSON", prefix.c_str(), 1);
  if (getenv("NCCL_SYM_KERNEL") == nullptr) setenv("NCCL_SYM_KERNEL", "AllGather_RailRing_LsaSTMC", 1);
  if (getenv("NCCL_SYM_GIN_KERNELS_ENABLE") == nullptr) setenv("NCCL_SYM_GIN_KERNELS_ENABLE", "1", 1);
  if (getenv("NCCL_SYM_CE_THRESHOLD") == nullptr) setenv("NCCL_SYM_CE_THRESHOLD", "1099511627776", 1);

  ncclUniqueId id;
  if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  void* sendbuff = nullptr;
  void* recvbuff = nullptr;
  NCCLCHECK(ncclMemAlloc(&sendbuff, sendBytes));
  NCCLCHECK(ncclMemAlloc(&recvbuff, recvBytes));
  CUDACHECK(cudaMemsetAsync(sendbuff, rank & 0xff, sendBytes, stream));
  CUDACHECK(cudaMemsetAsync(recvbuff, 0, recvBytes, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  ncclWindow_t sendWin;
  ncclWindow_t recvWin;
  NCCLCHECK(ncclCommWindowRegister(comm, sendbuff, sendBytes, &sendWin, NCCL_WIN_COLL_SYMMETRIC));
  NCCLCHECK(ncclCommWindowRegister(comm, recvbuff, recvBytes, &recvWin, NCCL_WIN_COLL_SYMMETRIC));

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < opt.warmup; i++) {
    NCCLCHECK(ncclAllGather(sendbuff, recvbuff, sendBytes, ncclUint8, comm, stream));
  }
  CUDACHECK(cudaStreamSynchronize(stream));

  double hostStart = MPI_Wtime();
  for (int i = 0; i < opt.iters; i++) {
    NCCLCHECK(ncclAllGather(sendbuff, recvbuff, sendBytes, ncclUint8, comm, stream));
  }
  CUDACHECK(cudaStreamSynchronize(stream));
  double hostEnd = MPI_Wtime();

  NCCLCHECK(ncclCommWindowDeregister(comm, sendWin));
  NCCLCHECK(ncclCommWindowDeregister(comm, recvWin));
  NCCLCHECK(ncclMemFree(sendbuff));
  NCCLCHECK(ncclMemFree(recvbuff));
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommDestroy(comm));

  // NCCL materializes profile JSON from stream callbacks. Some callbacks are
  // drained during communicator teardown, so parse after ncclCommDestroy().
  std::vector<double> local;
  for (int retry = 0; retry < 200; retry++) {
    local = parseBodyTimesUs(opt.profileDir, opt.warmup, opt.iters, opt.ticksPerUs);
    if ((int)local.size() >= opt.iters) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!opt.keepProfile) cleanupProfileDir(opt.profileDir);

  int localCount = (int)local.size();
  std::vector<int> counts(nranks), displs(nranks);
  MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    for (int i = 1; i < nranks; i++) displs[i] = displs[i - 1] + counts[i - 1];
  }
  int totalCount = 0;
  if (rank == 0) totalCount = std::accumulate(counts.begin(), counts.end(), 0);
  std::vector<double> all(totalCount);
  MPI_Gatherv(local.data(), localCount, MPI_DOUBLE,
              all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double localHostAvgUs = (hostEnd - hostStart) * 1.0e6 / std::max(1, opt.iters);
  double maxHostAvgUs = 0.0;
  MPI_Reduce(&localHostAvgUs, &maxHostAvgUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::sort(all.begin(), all.end());
    double avg = all.empty() ? 0.0 : std::accumulate(all.begin(), all.end(), 0.0) / all.size();
    printf("# bodytime_nccl_test ranks=%d recv_bytes=%zu send_bytes=%zu warmup=%d iters=%d samples=%zu ticks_per_us=%.3f\n",
           nranks, recvBytes, sendBytes, opt.warmup, opt.iters, all.size(), opt.ticksPerUs);
    printf("# host_max_avg_us %.3f\n", maxHostAvgUs);
    printf("metric,us\n");
    printf("avg,%.3f\n", avg);
    printf("min,%.3f\n", all.empty() ? 0.0 : all.front());
    printf("p50,%.3f\n", percentile(all, 0.50));
    printf("p90,%.3f\n", percentile(all, 0.90));
    printf("p95,%.3f\n", percentile(all, 0.95));
    printf("p99,%.3f\n", percentile(all, 0.99));
    printf("max,%.3f\n", all.empty() ? 0.0 : all.back());
  }

  MPI_Finalize();
  return 0;
}
