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

        // CPU����ǰ׺��
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
            // TODO: CPUǰ׺��ʵ��
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
            // ��ѹ�������������б�Ƿ�0��Ԫ��->�Ա���󲿷ֺͣ�������ֺͱ仯������
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
            // 1. �����Ҫ������Ԫ��
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
                keep[i] = idata[i] ? 1 : 0;
            // 2. �Ա����ǰ׺�ͣ�scan���鱣��ǰ׺�� -> ��scan[i]���ӵ���һ��λ�ã���keep[i]Ϊ�棬��idata[i]�ŵ�odata[scan[i]]
            calc_prefix_sum(n, scan, keep);
            // 3. ����ǰ׺�ͣ�����idata�Ķ�ӦԪ�ص�odata
#pragma omp parallel for
            for (int i = 0; i < n; ++i)
                if (keep[i])
                    odata[scan[i]] = idata[i];
            // 4. ����Ԫ�ظ����������˶��ٸ�Ԫ�� + scan���һ��Ԫ���Ƿ��� ? -> scan[n-1]+keep[n-1]
            int cnt = scan[n - 1] + keep[n - 1];
            timer().endCpuTimer();
            delete[] keep;
            delete[] scan;
            return cnt;
        }
    }
}
