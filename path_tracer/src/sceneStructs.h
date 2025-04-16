#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "glm/gtx/intersect.hpp"
#include <stb_image.h>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    // Mesh:
    int tri_start_idx;
    int tri_end_idx;
    int boundingbox_idx;
    glm::vec3 min_bound;
    glm::vec3 max_bound;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    // 自定义纹理:
    int texture_id{ -1 };
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    int bounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  // 纹理坐标，两个维度取值范围都是[0,1]
  glm::vec2 texture_coord;
};

// 自定义的数据结构
#include "boundingbox.h"
class Triangle
{
public:
    int id;
    glm::vec3 vertices[3];
    glm::vec2 vertices_texture_coord[3];
    glm::vec3 vertex_normals[3];
    struct {
        glm::vec3 min_corner;
        glm::vec3 max_corner;
    } BoundBox;

    Triangle() {}

    __host__ __device__
    glm::vec3 get_bbox_center() const
    {
        return (BoundBox.min_corner + BoundBox.max_corner) / 2.0f;
    }
    __host__ __device__
    float area()
    {
        return glm::length(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
    }
    __host__ __device__
    void calculate_boundaries(glm::vec3& min, glm::vec3& max)
    {
        min.x = glm::min(glm::min(vertices[0].x, vertices[1].x), vertices[2].x);
        min.y = glm::min(glm::min(vertices[0].y, vertices[1].y), vertices[2].y);
        min.z = glm::min(glm::min(vertices[0].z, vertices[1].z), vertices[2].z);
        max.x = glm::max(glm::max(vertices[0].x, vertices[1].x), vertices[2].x);
        max.y = glm::max(glm::max(vertices[0].y, vertices[1].y), vertices[2].y);
        max.z = glm::max(glm::max(vertices[0].z, vertices[1].z), vertices[2].z);
    }
    __host__ __device__
    BoundingBox getBoundingBox()
    {
        glm::vec3 min(0.0f), max(0.0f);
        calculate_boundaries(min, max);
        return BoundingBox(min, max);
    }

    __host__ __device__
    bool intersect(const Ray& r, ShadeableIntersection& i)
    {
        glm::vec3 barypos(0.0f);
        if (glm::intersectRayTriangle(r.origin, r.direction,
            vertices[0], vertices[1], vertices[2], barypos)) {
            i.t = barypos.z;
            i.texture_coord = vertices_texture_coord[0] * (1.0f - barypos.x - barypos.y) + vertices_texture_coord[1] * barypos.x + vertices_texture_coord[2] * barypos.y;
            i.surfaceNormal = vertex_normals[0] * barypos.x + vertex_normals[1] * barypos.y + vertex_normals[2] * (1.0f - barypos.x - barypos.y);
            i.surfaceNormal = glm::normalize(i.surfaceNormal);
            return true;
        } else {
            i.t = -1.0f;
            return false;
        }
    }
};
class Texture {
public:
    int id{ -1 };
    int width{ -1 };
    int height{ -1 };
    int components{ -1 };
    unsigned char* image{ NULL };
    unsigned char* dev_image{ NULL };

    bool load(const char* filename) {
        if (image = stbi_load(filename, &width, &height, &components, 0))
            return true;
        return false;
    }
    __host__ __device__ glm::vec3 get_color(glm::vec2& texture_coord)
    {
        int X = glm::min(1.f * width * texture_coord.x, 1.f * width - 1.0f);
        int Y = glm::min(1.f * height * (1.0f - texture_coord.y), 1.f * height - 1.0f);
        int texid = Y * width + X;
        if (components == 3) {
            glm::vec3 col = glm::vec3(dev_image[texid * components],
                dev_image[texid * components + 1],
                dev_image[texid * components + 2]);
            col = 0.003921568627f * col;
            return col;
        }
        return glm::vec3(0.0f);
    }
};