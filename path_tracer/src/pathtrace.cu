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

// (��ImgUI�ؼ����õ�)ѡ��״̬
class Settings {
public:
    bool enable_RussianRoulette;
    bool enable_SortbyMaterial;
    bool use_Thrust;
    bool use_BoundingBoxTest;
    bool use_BVHtree;
    bool enable_SSAA;
    bool enable_DoF;
    float focal_dist;
    float aperture;
};
// ����ѡ��
void set(Settings& settings, GuiDataContainer* guidata)
{
    settings.enable_RussianRoulette = guidata->russianRoulette;
    settings.enable_SortbyMaterial = guidata->sortbyMaterial;
    settings.use_Thrust = guidata->useThrustPartition;
    settings.use_BVHtree = guidata->useBVHtree;
    settings.use_BoundingBoxTest = guidata->useBBox;
    settings.enable_SSAA = guidata->SSAA;
    settings.enable_DoF = guidata->DoF;
    settings.aperture = guidata->aperture;
    settings.focal_dist = guidata->focal_len;
}

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
// faster stream compaction
static PathSegment* dev_paths_buf = NULL;
static int* path_boolean_buf = NULL;
static int* path_indices_buf = NULL;
// Mesh & BVH
static Triangle* dev_triangles = NULL;
static BoundingBox* dev_boundingboxes = NULL;
static BVHTree* dev_bvh_tree = NULL;
// Texture Mapping
static Texture* dev_textures = NULL;

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
    cudaMalloc(&path_boolean_buf, pixelcount * sizeof(int));
    cudaMalloc(&path_indices_buf, pixelcount * sizeof(int));
    // Bounding boxes & BVH trees
    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_boundingboxes, scene->bounding_boxes.size() * sizeof(BoundingBox));
    cudaMemcpy(dev_boundingboxes, scene->bounding_boxes.data(), scene->bounding_boxes.size() * sizeof(BoundingBox), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_bvh_tree, scene->n_bvh_nodes * sizeof(BVHTree));
    cudaMemcpy(dev_bvh_tree, scene->bvh_tree, scene->n_bvh_nodes * sizeof(BVHTree), cudaMemcpyHostToDevice);
    // Textures
    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
    for (auto& texture:scene->textures) {
        auto [ _, w, h, n_com, __,___ ] = texture;
        int n_bytes = w * h * n_com;
        cudaMalloc(&texture.dev_image, n_bytes * sizeof(unsigned char));
        cudaMemcpy(texture.dev_image, texture.image, n_bytes * sizeof(unsigned char), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

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
    cudaFree(path_boolean_buf);
    cudaFree(path_indices_buf);
    // Bounding boxes & BVH trees
    cudaFree(dev_triangles);
    cudaFree(dev_boundingboxes);
    cudaFree(dev_textures);
    cudaFree(dev_bvh_tree);
    checkCUDAError("pathtraceFree");
}

// ���ɴӹ۲��߾�ͷ�����Ĺ���·��(��1�η���)
// �����ڴ�ʵ�֣�
//      ����� - ���ɴ����ز����Ĺ���
//      ��̬ģ�� - ��ʱ�Ŷ�����
//      ��ͷЧ�� - ���ݾ�ͷ�Ŷ��������
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, Settings* settings)
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

    // ������������
    if (settings->enable_SSAA) {
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng))
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng))
        );
    } else {
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
    }
    if (settings->enable_DoF) {
        // ������ͷ����
        float r = u01(rng) * settings->aperture;
        float theta = u01(rng) * 2 * PI;
        glm::vec3 p_lens(r * cos(theta), r * sin(theta), 0.0f);
        // ���㽹ƽ��
        float ft = settings->focal_dist / glm::abs(segment.ray.direction.z);
        glm::vec3 p_focus = segment.ray.origin + ft * segment.ray.direction;
        // �������
        segment.ray.origin += p_lens;
        segment.ray.direction = glm::normalize(p_focus - segment.ray.origin);
    }
    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;   
    segment.bounces = 0;
}

// ����·������
__global__ void computeIntersections( int depth, int num_paths, PathSegment* pathSegments,  Geom* geoms, int geoms_size, 
    ShadeableIntersection* intersections, Triangle* triangles, BoundingBox* bounds, BVHTree* bvh_tree, Settings* settings)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index >= num_paths)
        return;
    PathSegment& pathSegment = pathSegments[path_index];

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    // coord��2ά�������꣬ȡֵΪ[0,1]
    glm::vec2 coord;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    int hit_tri_index = -1;
    bool outside = true;
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec2 tmp_coord;

    // naive parse through global geoms
    for (int i = 0; i < geoms_size; ++i){
        Geom& geom = geoms[i];
        if (geom.type == CUBE)
            t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        else if (geom.type == SPHERE)
            t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        // TODO: add more intersection tests here... triangle? metaball? CSG?
        // ��ײ��⣺����
        else if (geom.type == MESH) {
            if (settings->use_BVHtree) {
                ShadeableIntersection is;
                is.t = FLT_MAX;
                t = -1.0f;
                if (BVHIntersectionTest(pathSegment.ray, hit_tri_index, is, bvh_tree, triangles)) {
                    if (hit_tri_index >= geom.tri_start_idx && hit_tri_index < geom.tri_end_idx) {
                        t = is.t;
                        tmp_coord = is.texture_coord;
                        tmp_normal = is.surfaceNormal;
                    }
                }
            } else {
                t = meshIntersectionTest(tmp_intersect, geom, pathSegment.ray, tmp_coord, tmp_normal, triangles, outside, settings->use_BoundingBoxTest);
            }
        }
        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.

        if (t > 0.0f && t_min > t){
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                coord = tmp_coord;
                normal = tmp_normal;
        }
    }

    if (hit_geom_index == -1){
        intersections[path_index].t = -1.0f;
    }else{
        // The ray hits something
        intersections[path_index].t = t_min;
        intersections[path_index].materialId = geoms[hit_geom_index].materialid;
        intersections[path_index].texture_coord = coord;
        intersections[path_index].surfaceNormal = normal;
    }
}

// ��ɫ��ʵ�֣�����BSDF
__global__ void shadeMaterials( int iter, int num_paths, Settings* settings, ShadeableIntersection* shadeableIntersections,
                                PathSegment* pathSegments, Material* materials, Texture* textures)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
        return;
    auto& path_segment = pathSegments[idx];
    auto& shade_intersect = shadeableIntersections[idx];
    if (shade_intersect.t <= 0.0f) {
        path_segment.color = glm::vec3(0.0f);
        path_segment.remainingBounces = 0;
        return;
    }
    thrust::uniform_real_distribution<float> u01(0, 1);
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, path_segment.remainingBounces);
    auto& material = materials[shade_intersect.materialId];

    if (material.emittance > 0.0f) {
        path_segment.color *= material.color * material.emittance;
        path_segment.remainingBounces = 0;
        return;
    }
    scatterRay(path_segment, getPointOnRay(path_segment.ray, shade_intersect.t), shade_intersect, material, textures, rng);
    path_segment.bounces += 1;
    if (--path_segment.remainingBounces == 0) {
        path_segment.color = glm::vec3(0.0f);
        return;
    }
    // ����˹�����㷨
    if (settings->enable_RussianRoulette && path_segment.bounces > 3) {
        const auto luma_vec = glm::vec3(0.2126, 0.7152, 0.0722);
        float segment_luma = glm::dot(path_segment.color, luma_vec);
        // ��ֹ���ʣ�ʹ��·���ĵ�ǰ��ɫ���ۻ������ʣ���������ֹ����
        float q = glm::max(0.05f, 1 - segment_luma);
        if (u01(rng) < q) {
            // c = 0
            path_segment.color = glm::vec3(0.0f);
            path_segment.remainingBounces = 0;
            return;
        }
        // F -> (F-qc)/(1-q) = F/1-q
        path_segment.color /= 1.0f - q;
    }
}

// �����ε������Ӧ�õ�ͼ����
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

// ȥ������Ҫ�Ĺ���·��
__global__ void mark_valid(int n_paths, int* dev_bools, PathSegment* dev_paths)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_paths)
        return;
    dev_bools[tid] = dev_paths[tid].remainingBounces == 0 ? 0 : 1;
}
__global__ void keep(int n_paths, int new_npaths, PathSegment* dev_out_paths, PathSegment* dev_paths, 
                     const int* bools, const int* indices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_paths)
        return;
    if (bools[tid]) // ֻҪbools[tid]Ϊ�棬indices[tid]������ͬ�������ͻ
        dev_out_paths[indices[tid]] = dev_paths[tid];
    else            // ����ЧԪ�أ�tid ������һ������ indices[tid]
        dev_out_paths[new_npaths + tid - indices[tid]] = dev_paths[tid];
}
int relocate_terminated_paths(int n_paths, dim3 n_blocks, int blocksize, PathSegment* dev_paths, 
                              PathSegment* buf, int* bools, int* indices)
{
    // Hack: ���⴦��ֻ��1�����ߵ�����
    if (n_paths == 1) {
        int bounces;
        cudaMemcpy(&bounces, &(dev_paths->remainingBounces), sizeof(int), cudaMemcpyDeviceToHost);
        return bounces == 0 ? 0 : 1;
    }
    // ���������� StreamCompaction ��ʵ��
    int copy_last, valid_elems, new_npaths;

    cudaStream_t copy_stream;
    cudaStreamCreate(&copy_stream);
    cudaEvent_t ready_event;
    cudaMemcpyAsync(buf, dev_paths, n_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice, copy_stream);
    cudaEventCreate(&ready_event);
    cudaEventRecord(ready_event, copy_stream);

    mark_valid<<<n_blocks,blocksize>>>(n_paths, bools, dev_paths);
    StreamCompaction::Efficient::scan(n_paths, indices, bools);
    cudaMemcpy(&valid_elems, indices + n_paths - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&copy_last, bools + n_paths - 1, sizeof(int), cudaMemcpyDeviceToHost);
    new_npaths = copy_last + valid_elems;
    cudaEventSynchronize(ready_event);
    keep<<<n_blocks, blocksize>>> (n_paths, new_npaths, dev_paths, buf, bools, indices);
   
    cudaEventDestroy(ready_event);
    cudaStreamDestroy(copy_stream);
    return new_npaths;
}

// ����������
struct material_compare {
    __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
        return a.materialId < b.materialId;
    }
};

struct is_valid {
	__device__ bool operator()(const PathSegment& path) {
		return path.remainingBounces > 0;
	}
};

// ��ں���
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

    Settings settings;
    set(settings, guiData);
    cudaMemcpy(dev_settings, &settings, sizeof(Settings), cudaMemcpyHostToDevice);

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, dev_settings);
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
        
        set(settings, guiData);

        cudaEvent_t ready_event;
        cudaMemcpyAsync(dev_settings, &settings, sizeof(Settings), cudaMemcpyHostToDevice, copy_stream);
        cudaEventCreate(&ready_event);
        cudaEventRecord(ready_event, copy_stream);

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        dim3 numblocksPathSegmentTracing = (num_remaining_paths + blockSize1d - 1) / blockSize1d;
        cudaEventSynchronize(ready_event); 
        cudaEventDestroy(ready_event);
        cudaStreamDestroy(copy_stream);

        // ����·������
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> ( depth, num_remaining_paths, dev_paths, dev_geoms, 
            hst_scene->geoms.size(), dev_intersections, dev_triangles, dev_boundingboxes, dev_bvh_tree, dev_settings );
        checkCUDAError("trace one bounce");
        depth += 1;
        cudaDeviceSynchronize();
        
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

        // ������ɫ��
        shadeMaterials<<<numblocksPathSegmentTracing, blockSize1d>>>( iter, num_remaining_paths, dev_settings,
            dev_intersections, dev_paths, dev_materials, dev_textures );
        checkCUDAError("shade materials");
        cudaDeviceSynchronize();
        // �Ƴ���ֹ��·��
        if(settings.use_Thrust)
            num_remaining_paths =   thrust::stable_partition(
                                        thrust::device, thrust::device_pointer_cast(dev_paths),
                                        thrust::device_pointer_cast(dev_paths + num_remaining_paths), is_valid()
                                    ) - thrust::device_pointer_cast(dev_paths);
        else
            num_remaining_paths =  relocate_terminated_paths(num_remaining_paths, numblocksPathSegmentTracing, blockSize1d, 
                                                            dev_paths, dev_paths_buf, path_boolean_buf, path_indices_buf);   
        
        checkCUDAError("relocate terminated paths");
        cudaDeviceSynchronize();
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
