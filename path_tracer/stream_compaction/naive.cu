#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__
        void gpu_naive_scan(int n, int offset, int* odata, const int* idata)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= n)
                return;
            odata[tid] = idata[tid];
            if (tid >= offset)
                odata[tid] += idata[tid - offset];
        }
        __global__
        void gpu_incl2excl_pfxsum(int n, int* odata, const int* idata)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= n)
                return;
            odata[tid] = tid == 0 ? 0 : idata[tid - 1];
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_idata, *dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            // ��ȡkernel��������
            cudaDeviceProp dev_prop;
            cudaGetDeviceProperties(&dev_prop, 0);
            int minBlockSize = dev_prop.warpSize,
                maxBlockSize = dev_prop.maxThreadsPerBlock;
            int blockSize = std::max(minBlockSize, std::min(maxBlockSize, n));
            int gridSize = (n + blockSize - 1) / blockSize;
            int max_exp = ilog2ceil(n); // offset: 2^(d-1)

            timer().startGpuTimer();
            // TODO
            // 1. ����kernel����ǰ׺��
            for (int d = 0; d < max_exp; ++d) {
                gpu_naive_scan <<< gridSize, blockSize >>> (n, 1<<d, dev_odata, dev_idata);
                // ԭ�ظ��µ��¾�̬��������Ҫswap
                std::swap(dev_odata, dev_idata);
            }
            // 2. Ϊ������ѹ���������󣬽�������ǰ׺��ת��Ϊ�ǰ�����
            gpu_incl2excl_pfxsum <<< gridSize, blockSize >>> (n, dev_odata, dev_idata);
            timer().endGpuTimer();

            // ���������������������
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
