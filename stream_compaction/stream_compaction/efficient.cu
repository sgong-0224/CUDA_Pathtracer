#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

/*
 *  ���ܵ������Ż���
 *  ��  ���ù����ڴ�
 *  ��  �����߳��������ٷ�֧��ɢ
 *  ��  ����ô�BANK��ַ��ͻ
 *  ��  �ô�ϲ���ÿ���̰߳��������ȡ
 *  TODO: ѭ��չ����#pragma unroll block_size
 */

#define CEIL_DIV(x,y)        (((x)+(y)-1)/(y))
int** buf = nullptr;

// ����BANK��ͻ�ĺ궨��
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n)>>NUM_BANKS + (n)>>(2*LOG_NUM_BANKS))
#define POSITION(x)             ((x)+CONFLICT_FREE_OFFSET(x))

// ƽ����
#define LCHILD(x)           ((x)<<1)
#define RCHILD(x)           (1+((x)<<1))
#define REDUCE_LCHILD(x)    (1+((x)<<1))
#define SCATTER_LCHILD(x)   (1+((x)<<1))
#define REDUCE_RCHILD(x)    (2+((x)<<1))
#define SCATTER_RCHILD(x)   (2+((x)<<1))

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /*
         *  kernel functions impl.
         */
        // TODO
        // 1. һ���߳̿��ڵ�ǰ׺��
        __global__
		void gpu_scan_block(int n, int* sum_buf, int* odata, const int* idata)
		{
            extern __shared__ int shm_data[];
            int blocksize = n < (blockDim.x<<1) ? n : (blockDim.x<<1);

            int tid        = threadIdx.x,
                init_l     = threadIdx.x,
                init_r     = threadIdx.x + blockDim.x,
                global_idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

            int stride = 1;
            // ������̲߳����ı��������������ڴ棬����������������ʼλ��
            shm_data[POSITION(init_l)] = global_idx < n ? idata[global_idx] : 0;
            shm_data[POSITION(init_r)] = global_idx + blockDim.x < n ? idata[global_idx + blockDim.x] : 0;
            // reduce
            for (int d = blocksize >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (tid < d) {
                    int l = POSITION(stride * REDUCE_LCHILD(tid) - 1);
                    int r = POSITION(stride * REDUCE_RCHILD(tid) - 1);
                    shm_data[r] += shm_data[l];
                }
                stride <<= 1;
            }
            if (tid == 0) {
                sum_buf[blockIdx.x] = shm_data[POSITION(blocksize-1)];
                shm_data[POSITION(blocksize - 1)] = 0;
            }
            // scatter
            for (int d = 1; d < blocksize; d <<= 1) {
                stride >>= 1;
                __syncthreads();
                if (tid < d) {
                    int l = POSITION(stride * REDUCE_LCHILD(tid) - 1);
                    int r = POSITION(stride * REDUCE_RCHILD(tid) - 1);
                    int tmp = shm_data[l];
                    shm_data[l] = shm_data[r];
                    shm_data[r] += tmp;
                }
            }
            // д�����λ��
            __syncthreads();
            if (global_idx < n)
                odata[global_idx] = shm_data[POSITION(init_l)];
            if (global_idx + blockDim.x < n)
                odata[global_idx + blockDim.x] = shm_data[POSITION(init_r)];
        }

        // 2. ���߳̿��ǰ׺�ͣ���ǰһ���߳̿����ͽ���ӵ����߳̿������Ԫ��
        __global__
        void gpu_prefix_add(int n, int* odata, const int* idata)
        {
            int block_sum = idata[blockIdx.x];
            int block_elems = blockDim.x * 2;
            int elem_idx = blockIdx.x * block_elems + threadIdx.x;
            if (elem_idx >= n)
                return;
            odata[elem_idx] += block_sum;
            if (elem_idx + blockDim.x < n )
                odata[elem_idx + blockDim.x] += block_sum;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        // ά��ǰ׺�͵��м����
        int allocate_sum_buf(int max_depth, int blocksize)
        {
            buf = (int**)malloc(max_depth * sizeof(int*));
            int elems_per_block = blocksize << 1;
            int blocks = CEIL_DIV((1 << max_depth), elems_per_block);
            int d = 0;
#pragma unroll 4
            for(d=0;d<max_depth;++d){
                cudaMalloc((void**)&buf[d], blocks * sizeof(int));
                if (blocks == 1)
                    break;
                blocks = CEIL_DIV(blocks, elems_per_block);
            }
            return d + 1;
        }
        void free_sum_buf(int max_depth)
        {
#pragma unroll 4
            for (int i = 0; i < max_depth; ++i)
                cudaFree(buf[i]);
            free(buf);
            buf = nullptr;
        }

        // �Գ���1�� block ��С���̣߳��ݹ�ؽ��� reduce - scatter �Լ���ǰ׺�ͣ���ʱ����ԭ�ؼ���
        void recursive_scan(int n, int blocksize, int *odata, const int *idata, int depth = 0)
        {
            int elems_per_block = blocksize << 1;
            // 1���̲߳���2��Ԫ��
            int blocks = CEIL_DIV(n, elems_per_block);
            gpu_scan_block <<< blocks, blocksize, elems_per_block * sizeof(int) >>> (n, buf[depth], odata, idata);
            
            if (blocks > 1) {
                recursive_scan(blocks, blocksize, buf[depth], buf[depth], depth + 1);
                gpu_prefix_add <<< blocks, blocksize >>> (n, odata, buf[depth]);
            }
        }
        
        // scan �������
        void scan(int n, int *odata, const int *idata) {
            int max_exp = ilog2ceil(n);
            int n_pad = 1 << max_exp; // ��䵽2����
            int *d_idata,*d_odata;
            cudaMalloc((void**)&d_idata, n_pad * sizeof(int));
            cudaMemset(d_idata, 0, n_pad * sizeof(int));
            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_odata, n_pad * sizeof(int));
            // ��ȡkernel��������
            cudaDeviceProp dev_prop;
            cudaGetDeviceProperties(&dev_prop, 0);
            int minBlockSize = dev_prop.warpSize;
            int blockSize = std::min(minBlockSize, n);
            max_exp = allocate_sum_buf(max_exp, blockSize);
            timer().startGpuTimer();
            // TODO
            recursive_scan(n_pad, blockSize, d_odata, d_idata);
            timer().endGpuTimer();

            cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            free_sum_buf(max_exp);
            cudaFree(d_idata);
            cudaFree(d_odata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int *keep, *scan_res, *dev_idata, *dev_odata;
            cudaMalloc((void**)&keep, n * sizeof(int));
            cudaMalloc((void**)&scan_res, n * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            // ��ȡkernel��������
            cudaDeviceProp dev_prop;
            cudaGetDeviceProperties(&dev_prop, 0);
            int minBlockSize = dev_prop.warpSize;
            int blockSize = std::min(minBlockSize, n);
            int gridSize = CEIL_DIV(n, blockSize);
            
            // timer().startGpuTimer();
            // TODO
            // 1. ������ЩԪ����Ҫ����
            StreamCompaction::Common::kernMapToBoolean <<< gridSize, blockSize >>> (n, keep, dev_idata);
            // 2. ��ǰ׺��
            scan(n, scan_res, keep);
            // 3. ����Ԫ��
            StreamCompaction::Common::kernScatter <<< gridSize, blockSize >>> (n, dev_odata, dev_idata, keep, scan_res);
            // timer().endGpuTimer();
            
            // 4. �����������ѹ�������Ԫ�ظ���, ����
            int cnt = 0, keep_last = 0;
            cudaMemcpy(&keep_last, keep + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&cnt, scan_res + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);
            cudaFree(scan_res);
            cudaFree(keep);
            cudaFree(dev_idata);

            return cnt+keep_last;
        }
    }
}
