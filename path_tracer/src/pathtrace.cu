#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "../stream_compaction/efficient.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

extern void checkCUDAErrorFn(const char* msg, const char* file, int line);

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
// ...

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
}

// 计算光线路径交叉
__global__ void computeIntersections( int depth, int num_paths, PathSegment* pathSegments, 
    Geom* geoms, int geoms_size, ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
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

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            else if (geom.type == SPHERE)
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// 着色器实现，处理BSDF: 出射光和入射光的亮度关系
__global__ void shadeMaterials(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
        return;
    auto& path_segment = pathSegments[idx];
    auto& shade_intersect = shadeableIntersections[idx];
    if (shade_intersect.t > 0.0f) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, path_segment.remainingBounces);
        auto& material = materials[shade_intersect.materialId];
        if (material.emittance > 0.0f) {
            path_segment.color *= material.color * material.emittance;
            path_segment.remainingBounces = 0;
        } else {
            scatterRay( path_segment, getPointOnRay(path_segment.ray, shade_intersect.t),
                        shade_intersect.surfaceNormal, material, rng);
            if (--path_segment.remainingBounces == 0)
                path_segment.color = glm::vec3(0.0f);
        }
    } else {
        path_segment.color = glm::vec3(0.0f);
        path_segment.remainingBounces = 0;
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

__global__ 
void mark_valid(int n_paths, int* dev_bools, PathSegment* dev_paths)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_paths)
        return;
    dev_bools[tid] = dev_paths[tid].remainingBounces ? 1 : 0;
}
__global__
void keep_valid(int n_paths, PathSegment* dev_out_paths, PathSegment* dev_paths, const int* bools, const int* indices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_paths)
        return;
    if (bools[tid])
        dev_out_paths[indices[tid]] = dev_paths[tid];
}
// FIXME: invalid mem access
int remove_terminated_paths(int n_paths, dim3 n_blocks, int blocksize, PathSegment* dev_paths)
{
    int new_npaths, *dev_bools, *dev_indices;
    cudaMalloc((void**)&dev_bools, n_paths * sizeof(int));
    cudaMalloc((void**)&dev_indices, n_paths * sizeof(int));
    mark_valid<<<n_blocks,blocksize>>>(n_paths, dev_bools, dev_paths);
    new_npaths = StreamCompaction::Efficient::compact(n_paths, dev_indices, dev_bools);
    keep_valid<<<n_blocks, blocksize>>>(n_paths, dev_paths, dev_paths, dev_bools, dev_indices);
    cudaFree(dev_bools);
    cudaFree(dev_indices);
    return new_npaths;
}

/*
// 移除终止路径
__global__
void reverse(int n, int* dev_bools)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    dev_bools[tid] = !dev_bools[tid];
}
__global__
void mark_invalid(int n_paths, int* dev_bools, PathSegment* dev_paths)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_paths)
        return;
    dev_bools[tid] = dev_paths[tid].remainingBounces ? 0 : 1;
}
__global__
void keep(int n_paths, PathSegment* dev_out_paths, PathSegment* dev_paths, const int* bools, const int* indices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_paths)
        return;
    auto path = dev_paths[tid];
    __syncthreads();
    if (bools[tid])
        dev_out_paths[indices[tid]] = path;
}
int relocate_terminated_paths(int n_paths, dim3 n_blocks, int blocksize, PathSegment* dev_paths)
{
    int n_termpaths, new_npaths;
    int* dev_bools, * dev_indices;
    PathSegment* invalid_paths;
    cudaMalloc((void**)&dev_bools, n_paths * sizeof(int));
    cudaMalloc((void**)&dev_indices, n_paths * sizeof(int));
    // 无效路径拷贝到 invalid_paths 备用
    mark_invalid<<<n_blocks,blocksize>>>(n_paths, dev_bools, dev_paths);
    n_termpaths = StreamCompaction::Efficient::compact(n_paths, dev_indices, dev_bools);
    cudaMalloc((void**)&invalid_paths, n_termpaths * sizeof(PathSegment));
    keep<<<n_blocks, blocksize>>>(n_paths, invalid_paths, dev_paths, dev_bools, dev_indices);
    // 有效路径原地计算
    reverse<<<n_blocks, blocksize>>> (n_paths, dev_bools);
    new_npaths = StreamCompaction::Efficient::compact(n_paths, dev_indices, dev_bools);
    keep<<<n_blocks, blocksize>>>(n_paths, dev_paths, dev_paths, dev_bools, dev_indices);
    // 整理路径
    cudaMemcpy(dev_paths + new_npaths, invalid_paths, n_termpaths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
    cudaFree(invalid_paths);
    cudaFree(dev_bools);
    cudaFree(dev_indices);
    return new_npaths;
}


*/

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
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        dim3 numblocksPathSegmentTracing = (num_remaining_paths + blockSize1d - 1) / blockSize1d;

        // 计算光线交叉
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> ( depth, 
            num_remaining_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth += 1;
        
        // 调用着色器
        shadeMaterials<<<numblocksPathSegmentTracing, blockSize1d>>>( iter, num_remaining_paths,
            dev_intersections, dev_paths, dev_materials );

        // TODO: 移除终止的路径
        num_remaining_paths = remove_terminated_paths(num_remaining_paths, numblocksPathSegmentTracing, blockSize1d, dev_paths);

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
