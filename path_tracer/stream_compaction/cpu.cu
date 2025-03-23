#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // CPU串行前缀和
        void calc_prefix_sum(int n, int* odata, const int* idata)
        {
            odata[0] = 0;
#pragma omp parallel for
            for (int i = 1; i < n; ++i)
                odata[i] = odata[i - 1] + idata[i - 1];
        }
        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO: CPU前缀和实现
            calc_prefix_sum(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            // 流压缩：保留数组中标记非0的元素->对标记求部分和，如果部分和变化，则保留
            int cnt = 0;
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
                if (idata[i])
                    odata[cnt++] = idata[i];
            timer().endCpuTimer();
            return cnt;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int *keep = new int[n], *scan = new int[n];
            timer().startCpuTimer();
            // TODO
            // 1. 标记需要保留的元素
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
                keep[i] = idata[i] ? 1 : 0;
            // 2. 对标记求前缀和，scan数组保存前缀和 -> 若scan[i]增加到下一个位置，且keep[i]为真，将idata[i]放到odata[scan[i]]
            calc_prefix_sum(n, scan, keep);
            // 3. 根据前缀和，保留idata的对应元素到odata
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
                if (keep[i])
                    odata[scan[i]] = idata[i];
            // 4. 计算元素个数：保留了多少个元素 + scan最后一个元素是否保留 ? -> scan[n-1]+keep[n-1]
            int cnt = scan[n - 1] + keep[n - 1];
            timer().endCpuTimer();
            delete[] keep;
            delete[] scan;
            return cnt;
        }
    }
}
