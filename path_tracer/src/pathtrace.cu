#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "../stream_compaction/efficient.h"
#include "../stream_compaction/naive.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

extern void checkCUDAErrorFn(const char* msg, const char* file, int line);

// (由ImgUI控件设置的)选项状态
class Settings {
public:
    bool enable_RussianRoulette;
    bool enable_SortbyMaterial;
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static Settings* dev_settings = NULL;
static PathSegment* dev_paths_buf = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_settings, sizeof(Settings));
    cudaMalloc(&dev_paths_buf, pixelcount * sizeof(PathSegment));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_settings);
    cudaFree(dev_paths_buf);

    checkCUDAError("pathtraceFree");
}

// 生成从观察者镜头出发的光线路径(第1次反射)
// 实现：
//      抗锯齿 - 生成次像素采样的光线
//      动态模糊 - 即时扰动光线
//      镜头效果 - 根据镜头扰动光线起点
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= cam.resolution.x || y >= cam.resolution.y)
        return;
    int index = x + (y * cam.resolution.x);
    PathSegment& segment = pathSegments[index];
    segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

    // TODO: implement antialiasing by jittering the ray
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
    thrust::uniform_real_distribution<float> u01(0, 1);

    segment.ray.direction = glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng))
        - cam.up    * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng))
    );

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;   
    segment.bounces = 0;
}

// 计算光线路径交叉
__global__ void computeIntersections( int depth, int num_paths, PathSegment* pathSegments, 
    Geom* geoms, int geoms_size, ShadeableIntersection* intersections, Settings* settings)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index >= num_paths)
        return;
    PathSegment pathSegment = pathSegments[path_index];

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    // naive parse through global geoms
    for (int i = 0; i < geoms_size; ++i){
        Geom& geom = geoms[i];
        if (geom.type == CUBE)
            t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        else if (geom.type == SPHERE)
            t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        // TODO: add more intersection tests here... triangle? metaball? CSG?

        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t_min > t){
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
        }
    }

    if (hit_geom_index == -1){
        intersections[path_index].t = -1.0f;
    }else{
        // The ray hits something
        intersections[path_index].t = t_min;
        intersections[path_index].materialId = geoms[hit_geom_index].materialid;
        intersections[path_index].surfaceNormal = normal;
    }
}

// 着色器实现，处理BSDF
__device__ void calc_segment(int iter, int idx, Settings* settings,  ShadeableIntersection& intersection, 
                             PathSegment& segment, Material* materials);
__global__ void shadeMaterials( int iter, int num_paths, Settings* settings, ShadeableIntersection* shadeableIntersections,
                                PathSegment* pathSegments, Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
        return;
    auto& path_segment = pathSegments[idx];
    auto& shade_intersect = shadeableIntersections[idx];
    calc_segment(iter, idx, settings, shade_intersect, path_segment, materials);
}
__device__ void calc_segment(int iter, int idx, Settings* settings, ShadeableIntersection& intersection,
                             PathSegment& segment, Material* materials)
{
    if (intersection.t <= 0.0f) {
        segment.color = glm::vec3(0.0f);
        segment.remainingBounces = 0;
        return;
    }
    thrust::uniform_real_distribution<float> u01(0, 1);
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.remainingBounces);
    auto& material = materials[intersection.materialId];
    if (material.emittance > 0.0f) {
        segment.color *= material.color * material.emittance;
        segment.remainingBounces = 0;
        return;
    }
    scatterRay(segment, getPointOnRay(segment.ray, intersection.t), intersection.surfaceNormal, material, rng);
    segment.bounces += 1;
    if (--segment.remainingBounces == 0) {
        segment.color = glm::vec3(0.0f);
        return;
    }
    // 俄罗斯轮盘算法
    if (settings->enable_RussianRoulette && segment.bounces > 3) {
        const auto luma_vec = glm::vec3(0.2126, 0.7152, 0.0722);
        float segment_luma = glm::dot(segment.color, luma_vec);
        // 终止概率：使用路径的当前颜色（累积反射率）来决定终止概率
        float q = glm::max(0.05f, 1 - segment_luma);
        if (u01(rng) < q) {
            // c = 0
            segment.color = glm::vec3(0.0f);
            segment.remainingBounces = 0;
            return;
        }
        // F -> (F-qc)/(1-q) = F/1-q
        segment.color /= 1.0f - q;
    }
}

// 将本次迭代结果应用到图像上
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

// 去除不需要的光线路径
__global__ void mark_valid(int n_paths, int* dev_bools, PathSegment* dev_paths)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_paths)
        return;
    dev_bools[tid] = dev_paths[tid].remainingBounces > 0 ? 1 : 0;
}
__global__ void mark_invalid(int n_paths, int* dev_bools, PathSegment* dev_paths)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_paths)
        return;
    dev_bools[tid] = dev_paths[tid].remainingBounces > 0 ? 0 : 1;
}
__global__ void reverse(int n, int* out_bools, int* in_bools)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    out_bools[tid] = in_bools[tid] == 0 ? 1 : 0;
}
__global__ void keep(int n_paths, PathSegment* dev_out_paths, PathSegment* dev_paths, const int* bools, const int* indices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_paths)
        return;
    if (bools[tid])
        // 只要bools[tid]为真，indices[tid]互不相同，不会冲突
        dev_out_paths[indices[tid]] = dev_paths[tid];
}
int relocate_terminated_paths(int n_paths, dim3 n_blocks, int blocksize, PathSegment* dev_paths, PathSegment* buf)
{
    // Hack: 特殊处理只有1条光线的情形
    if (n_paths == 1) {
        int bounces;
        cudaMemcpy(&bounces, &(dev_paths->remainingBounces), sizeof(int), cudaMemcpyDeviceToHost);
        if (bounces == 0)
            return 0;
        return 1;
    }
    // 其他情形用 StreamCompaction 的实现
    int *keep_bools, *keep_indices, *drop_bools, *drop_indices;
    int copy_last, valid_elems, new_npaths, drop_npaths;
    cudaMalloc(&keep_bools, n_paths * sizeof(int));
    cudaMalloc(&drop_bools, n_paths * sizeof(int));
    cudaMalloc(&keep_indices, n_paths * sizeof(int));
    cudaMalloc(&drop_indices, n_paths * sizeof(int));

    mark_valid<<<n_blocks,blocksize>>>(n_paths, keep_bools, dev_paths);
    StreamCompaction::Efficient::scan(n_paths, keep_indices, keep_bools);
    mark_invalid << <n_blocks, blocksize >> > (n_paths, drop_bools, dev_paths);
    StreamCompaction::Efficient::scan(n_paths, drop_indices, drop_bools);
    cudaMemcpy(&valid_elems, keep_indices + n_paths - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&copy_last, keep_bools + n_paths - 1, sizeof(int), cudaMemcpyDeviceToHost);
    new_npaths = copy_last + valid_elems;
    keep<<<n_blocks, blocksize>>>(n_paths, buf, dev_paths, keep_bools, keep_indices);
    keep<<<n_blocks, blocksize>>> (n_paths, buf + new_npaths, dev_paths, drop_bools, drop_indices);
    // FIXED:
    cudaMemcpy(dev_paths, buf, n_paths*sizeof(PathSegment), cudaMemcpyDeviceToDevice);
    
    cudaFree(keep_bools);
    cudaFree(drop_bools);
    cudaFree(keep_indices); 
    cudaFree(drop_indices);
    return new_npaths;
}

// 按材质排序
struct material_compare {
    __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
        return a.materialId < b.materialId;
    }
};

// 判断路径有效
struct is_valid {
    __device__ bool operator()(const PathSegment& p) {
        return p.remainingBounces > 0;
    }
};
struct is_terminated {
    __device__ bool operator()(const PathSegment& p) {
        return p.remainingBounces <= 0;
    }
};

// 入口函数
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    int num_remaining_paths = num_paths;
    while (num_remaining_paths != 0)
    {
        cudaStream_t copy_stream;
        cudaStreamCreate(&copy_stream);

        Settings settings;
        settings.enable_RussianRoulette = guiData->russianRoulette;
        settings.enable_SortbyMaterial = guiData->sortbyMaterial;

        cudaMemcpyAsync(dev_settings, &settings, sizeof(Settings), cudaMemcpyHostToDevice, copy_stream);
        cudaEvent_t ready_event;
        cudaEventCreate(&ready_event);
        cudaEventRecord(ready_event, copy_stream);

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        dim3 numblocksPathSegmentTracing = (num_remaining_paths + blockSize1d - 1) / blockSize1d;
        cudaEventSynchronize(ready_event); 
        cudaEventDestroy(ready_event);
        cudaStreamDestroy(copy_stream);

        // 计算光线交叉
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> ( depth, num_remaining_paths, dev_paths, 
            dev_geoms, hst_scene->geoms.size(), dev_intersections, dev_settings );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth += 1;
        
        if (settings.enable_SortbyMaterial) {
            thrust::sort_by_key(
                thrust::device,
                // key: material
                thrust::device_pointer_cast(dev_intersections),
                thrust::device_pointer_cast(dev_intersections + num_remaining_paths),
                // value: paths
                thrust::device_pointer_cast(dev_paths),
                // compare ordering
                material_compare()
            );
            checkCUDAError("sort intersections by material");
        }

        // 调用着色器
        shadeMaterials<<<numblocksPathSegmentTracing, blockSize1d>>>( iter, num_remaining_paths, dev_settings,
            dev_intersections, dev_paths, dev_materials );
        checkCUDAError("shade materials");

        // 移除终止的路径
        num_remaining_paths = relocate_terminated_paths(num_remaining_paths, numblocksPathSegmentTracing, blockSize1d, 
                                                        dev_paths, dev_paths_buf);
		
        /* thrust implementation: 
            num_remaining_paths = thrust::stable_partition(
                thrust::device,
                thrust::device_pointer_cast(dev_paths),
			    thrust::device_pointer_cast(dev_paths + num_remaining_paths),
                is_valid()
            ) - thrust::device_pointer_cast(dev_paths);
        */
        
        checkCUDAError("relocate terminated paths");

        if (guiData != NULL)
            guiData->TracedDepth = depth;
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
