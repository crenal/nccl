/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#include "kernel.cuh"
#include "primitives.cuh"
#include "gin_scratch__types.h"

#if defined(NCCL_SYM_AG_GIN_PROFILE)
static __device__ __forceinline__ uint64_t ncclSymkAgGinGlobalTimer() {
  uint64_t timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
}

static __device__ __forceinline__ void ncclSymkAgGinRecordEvent(
    ncclSymkDevWorkArgs const* args, int path, int eventIndex, int eventType, int opIndex,
    uint64_t startCycles, uint64_t elapsedCycles, uint64_t bytes, uint64_t offset, uint64_t signalValue,
    int step, int dataPeer, int worldRank, ncclDevComm const& comm, ncclTeam rail, int ginContext) {
  if (args->agGinProfile == nullptr || args->agGinProfileEventsPerPath <= 0) return;

  int eventsPerPath = args->agGinProfileEventsPerPath;
  int slotEventIndex = eventIndex;
  int slotEventType = eventType;
  if (eventIndex >= eventsPerPath - 1) {
    if (eventIndex != eventsPerPath - 1) return;
    slotEventIndex = eventsPerPath - 1;
    slotEventType = ncclSymkAgGinProfileEventOverflow;
  }

  int record = ((int)blockIdx.x * ncclSymkAgGinProfilePaths + path) * eventsPerPath + slotEventIndex;
  if (record >= args->agGinProfileRecords) return;

  ncclSymkAgGinProfileRecord prof = {};
  prof.startCycles = startCycles;
  prof.elapsedCycles = elapsedCycles;
  prof.bytes = bytes;
  prof.offset = offset;
  prof.signalValue = signalValue;
  prof.rank = comm.rank;
  prof.nRanks = comm.nRanks;
  prof.railRank = rail.rank;
  prof.railNRanks = rail.nRanks;
  prof.block = (int)blockIdx.x;
  prof.ginContext = ginContext;
  prof.path = path;
  prof.eventType = slotEventType;
  prof.eventIndex = eventIndex;
  prof.opIndex = opIndex;
  prof.step = step;
  prof.dataPeer = dataPeer;
  prof.worldRank = worldRank;
  args->agGinProfile[record] = prof;
}
#endif

__device__ __forceinline__ void ncclSymkRun_AllGather_RailRing_LsaSTMC(struct ncclSymkDevWorkArgs const* args) {
  ncclCoopCta cta;
  ncclSymkArgsHandler handler(args);
  ncclTeam rail = ncclTeamRail(handler.comm);
  int ginContext = (int)(blockIdx.x % handler.comm.ginContextCount);
  ncclGin gin(handler.comm, ginContext);
  constexpr int chunkSize = ncclSymkAllGather_RailRing_ChunkSize;
  ncclGinSignal_t railSignals = handler.ginSyncHandle.railSignals + blockIdx.x * rail.nRanks;
  ncclBarrierSession<ncclCoopCta> bar(cta, ncclTeamTagWorld(), gin, blockIdx.x, /*multimem=*/true);
  int nextPeer = (rail.rank + 1) % rail.nRanks;
  int prevPeer = (rail.rank + rail.nRanks - 1) % rail.nRanks;
  uint64_t* localSignalPtr = gin.getSignalShadowPtr(railSignals + prevPeer);
  uint64_t localSignalValue = *localSignalPtr;
  const int ringThreads = WARP_SIZE;

#if defined(NCCL_SYM_AG_GIN_PROFILE)
  int ringEventIndex = 0;
  int lsaEventIndex = 0;
  int ringPutIndex = 0;
  int ringWaitIndex = 0;
  int lsaWaitIndex = 0;
  int lsaBcastIndex = 0;
  uint64_t profileInitialBarrierStart = ncclSymkAgGinGlobalTimer();
#endif
  bar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::None);
#if defined(NCCL_SYM_AG_GIN_PROFILE)
  uint64_t profileInitialBarrierEnd = ncclSymkAgGinGlobalTimer();
  if (threadIdx.x == 0) {
    ncclSymkAgGinRecordEvent(args, ncclSymkAgGinProfilePathRing, ringEventIndex++,
      ncclSymkAgGinProfileEventInitialBarrier, 0,
      profileInitialBarrierStart, profileInitialBarrierEnd - profileInitialBarrierStart,
      0, 0, 0, -1, -1, -1, handler.comm, rail, ginContext);
  }
  if (threadIdx.x == ringThreads) {
    ncclSymkAgGinRecordEvent(args, ncclSymkAgGinProfilePathLsa, lsaEventIndex++,
      ncclSymkAgGinProfileEventInitialBarrier, 0,
      profileInitialBarrierStart, profileInitialBarrierEnd - profileInitialBarrierStart,
      0, 0, 0, -1, -1, -1, handler.comm, rail, ginContext);
  }
#endif

  handler.template forEachWorkNoFusion<uint8_t>(
    [&]__device__(size_t nElts, size_t nAllElts, ncclSymPtr<uint8_t> input, ncclSymPtr<uint8_t> output) {
      if (threadIdx.x < ringThreads) {
        ncclCoopWarpSpan warps(0, 1, 0);
        for (int step = 0; step < rail.nRanks - 1; step++) {
          int dataPeer = (rail.rank - step + rail.nRanks) % rail.nRanks;
          int dgrank = ncclTeamRankToWorld(handler.comm, rail, dataPeer);
          size_t remainingElts = nElts;
          size_t offset = 0;
          if (dataPeer == rail.rank) {
            while (remainingElts) {
              size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Send data chunk to next peer in ring
#if defined(NCCL_SYM_AG_GIN_PROFILE)
              uint64_t t0 = 0;
              if (warps.thread_rank() == 0) t0 = ncclSymkAgGinGlobalTimer();
#endif
              gin.put(rail, nextPeer, output + dgrank * nAllElts + offset,
                input + offset, chunkElts,
                ncclGin_SignalInc{ railSignals + rail.rank }, ncclGin_None{}, warps);
#if defined(NCCL_SYM_AG_GIN_PROFILE)
              if (warps.thread_rank() == 0) {
                uint64_t t1 = ncclSymkAgGinGlobalTimer();
                ncclSymkAgGinRecordEvent(args, ncclSymkAgGinProfilePathRing, ringEventIndex++,
                  ncclSymkAgGinProfileEventPutSelf, ringPutIndex++,
                  t0, t1 - t0, chunkElts, offset, 0, step, dataPeer, dgrank, handler.comm, rail, ginContext);
              }
#endif
              offset += chunkElts;
              remainingElts -= chunkElts;
            }
          } else {
            while (remainingElts) {
              size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Wait for ready signal from next peer before sending
#if defined(NCCL_SYM_AG_GIN_PROFILE)
              uint64_t waitStart = 0;
              if (warps.thread_rank() == 0) waitStart = ncclSymkAgGinGlobalTimer();
#endif
              gin.waitSignal(warps, railSignals + prevPeer, localSignalValue + 1, 32);
#if defined(NCCL_SYM_AG_GIN_PROFILE)
              if (warps.thread_rank() == 0) {
                uint64_t waitEnd = ncclSymkAgGinGlobalTimer();
                ncclSymkAgGinRecordEvent(args, ncclSymkAgGinProfilePathRing, ringEventIndex++,
                  ncclSymkAgGinProfileEventWaitSignal, ringWaitIndex++,
                  waitStart, waitEnd - waitStart, 0, offset, localSignalValue + 1,
                  step, dataPeer, dgrank, handler.comm, rail, ginContext);
              }
#endif
              // Send data chunk to next peer in ring
#if defined(NCCL_SYM_AG_GIN_PROFILE)
              uint64_t putStart = 0;
              if (warps.thread_rank() == 0) putStart = ncclSymkAgGinGlobalTimer();
#endif
              gin.put(rail, nextPeer, output + dgrank * nAllElts + offset,
                output + dgrank * nAllElts + offset, chunkElts,
                ncclGin_SignalInc{ railSignals + rail.rank }, ncclGin_None{}, warps);
#if defined(NCCL_SYM_AG_GIN_PROFILE)
              if (warps.thread_rank() == 0) {
                uint64_t putEnd = ncclSymkAgGinGlobalTimer();
                ncclSymkAgGinRecordEvent(args, ncclSymkAgGinProfilePathRing, ringEventIndex++,
                  ncclSymkAgGinProfileEventPutRemote, ringPutIndex++,
                  putStart, putEnd - putStart, chunkElts, offset, 0,
                  step, dataPeer, dgrank, handler.comm, rail, ginContext);
              }
#endif
              offset += chunkElts;
              remainingElts -= chunkElts;
              localSignalValue++;
            }
          }
        }
#if defined(NCCL_SYM_AG_GIN_PROFILE)
        uint64_t flushStart = 0;
        if (warps.thread_rank() == 0) flushStart = ncclSymkAgGinGlobalTimer();
#endif
        gin.flush(warps);
#if defined(NCCL_SYM_AG_GIN_PROFILE)
        if (warps.thread_rank() == 0) {
          uint64_t flushEnd = ncclSymkAgGinGlobalTimer();
          ncclSymkAgGinRecordEvent(args, ncclSymkAgGinProfilePathRing, ringEventIndex++,
            ncclSymkAgGinProfileEventFlush, 0,
            flushStart, flushEnd - flushStart, 0, 0, 0, -1, -1, -1, handler.comm, rail, ginContext);
        }
#endif
      } else {
        ncclCoopWarpSpan warps(1, blockDim.x / WARP_SIZE - 1, 1);
        // Loop through rail ranks starting from itself
        for (int step = 0; step < rail.nRanks; step++) {
          int dataPeer = (rail.rank - step + rail.nRanks) % rail.nRanks;
          int dgrank = ncclTeamRankToWorld(handler.comm, rail, dataPeer);
          size_t remainingElts = nElts;
          size_t offset = 0;
          if (dataPeer == rail.rank) {
            while (remainingElts) {
              size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Put self rank's data
#if defined(NCCL_SYM_AG_GIN_PROFILE)
              uint64_t t0 = 0;
              if (threadIdx.x == ringThreads) t0 = ncclSymkAgGinGlobalTimer();
#endif
              bcastMultimem(handler, warps.num_threads(), warps.thread_rank(), input + offset, output + dgrank * nAllElts + offset, chunkElts);
#if defined(NCCL_SYM_AG_GIN_PROFILE)
              if (threadIdx.x == ringThreads) {
                uint64_t t1 = ncclSymkAgGinGlobalTimer();
                ncclSymkAgGinRecordEvent(args, ncclSymkAgGinProfilePathLsa, lsaEventIndex++,
                  ncclSymkAgGinProfileEventBcastSelf, lsaBcastIndex++,
                  t0, t1 - t0, chunkElts, offset, 0, step, dataPeer, dgrank, handler.comm, rail, ginContext);
              }
#endif
              offset += chunkElts;
              remainingElts -= chunkElts;
            }
          } else {
            while (remainingElts) {
              size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Wait for signal from other peers before putting their data
#if defined(NCCL_SYM_AG_GIN_PROFILE)
              uint64_t waitStart = 0;
              if (threadIdx.x == ringThreads) waitStart = ncclSymkAgGinGlobalTimer();
#endif
              gin.waitSignal(warps, railSignals + prevPeer, localSignalValue + 1, 32);
#if defined(NCCL_SYM_AG_GIN_PROFILE)
              if (threadIdx.x == ringThreads) {
                uint64_t waitEnd = ncclSymkAgGinGlobalTimer();
                ncclSymkAgGinRecordEvent(args, ncclSymkAgGinProfilePathLsa, lsaEventIndex++,
                  ncclSymkAgGinProfileEventWaitSignal, lsaWaitIndex++,
                  waitStart, waitEnd - waitStart, 0, offset, localSignalValue + 1,
                  step, dataPeer, dgrank, handler.comm, rail, ginContext);
              }
#endif
#if defined(NCCL_SYM_AG_GIN_PROFILE)
              uint64_t bcastStart = 0;
              if (threadIdx.x == ringThreads) bcastStart = ncclSymkAgGinGlobalTimer();
#endif
              bcastMultimem(handler, warps.num_threads(), warps.thread_rank(), output + dgrank * nAllElts + offset, output + dgrank * nAllElts + offset, chunkElts);
#if defined(NCCL_SYM_AG_GIN_PROFILE)
              if (threadIdx.x == ringThreads) {
                uint64_t bcastEnd = ncclSymkAgGinGlobalTimer();
                ncclSymkAgGinRecordEvent(args, ncclSymkAgGinProfilePathLsa, lsaEventIndex++,
                  ncclSymkAgGinProfileEventBcastRemote, lsaBcastIndex++,
                  bcastStart, bcastEnd - bcastStart, chunkElts, offset, 0,
                  step, dataPeer, dgrank, handler.comm, rail, ginContext);
              }
#endif
              offset += chunkElts;
              remainingElts -= chunkElts;
              localSignalValue++;
            }
          }
        }
      }
    }
  );

  // update the shadow signal value
  if (threadIdx.x == ringThreads) {
#if defined(NCCL_SYM_AG_GIN_PROFILE)
    uint64_t profileShadowSignalStart = ncclSymkAgGinGlobalTimer();
#endif
    *localSignalPtr = localSignalValue;
#if defined(NCCL_SYM_AG_GIN_PROFILE)
    uint64_t profileShadowSignalEnd = ncclSymkAgGinGlobalTimer();
    ncclSymkAgGinRecordEvent(args, ncclSymkAgGinProfilePathLsa, lsaEventIndex++,
      ncclSymkAgGinProfileEventShadowSignal, 0,
      profileShadowSignalStart, profileShadowSignalEnd - profileShadowSignalStart,
      0, 0, localSignalValue, -1, -1, -1, handler.comm, rail, ginContext);
#endif
  }
#if defined(NCCL_SYM_AG_GIN_PROFILE)
  uint64_t profileFinalBarrierStart = ncclSymkAgGinGlobalTimer();
#endif
  bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::None);
#if defined(NCCL_SYM_AG_GIN_PROFILE)
  uint64_t profileFinalBarrierEnd = ncclSymkAgGinGlobalTimer();
  if (threadIdx.x == 0) {
    ncclSymkAgGinRecordEvent(args, ncclSymkAgGinProfilePathRing, ringEventIndex++,
      ncclSymkAgGinProfileEventFinalBarrier, 0,
      profileFinalBarrierStart, profileFinalBarrierEnd - profileFinalBarrierStart,
      0, 0, 0, -1, -1, -1, handler.comm, rail, ginContext);
  }
  if (threadIdx.x == ringThreads) {
    ncclSymkAgGinRecordEvent(args, ncclSymkAgGinProfilePathLsa, lsaEventIndex++,
      ncclSymkAgGinProfileEventFinalBarrier, 0,
      profileFinalBarrierStart, profileFinalBarrierEnd - profileFinalBarrierStart,
      0, 0, 0, -1, -1, -1, handler.comm, rail, ginContext);
  }
#endif
}
