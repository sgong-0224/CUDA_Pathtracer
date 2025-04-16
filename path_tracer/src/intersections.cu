#include "intersections.h"

__host__ __device__ 
float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ 
float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

// 其他形状的碰撞检测:
// 网格: 遍历
__host__ __device__ 
float meshIntersectionTest(
    glm::vec3& intersection_point, Geom mesh, Ray r, 
    glm::vec2& texture_coord, glm::vec3& normal, 
    Triangle* triangles, bool& from_outside,
    bool use_boundingbox
)
{
    // 计算mesh上的光线
    glm::vec3 ray_origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 ray_direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));
    Ray ray{ ray_origin,ray_direction };

    if ( use_boundingbox ) {
        // 快速检测，和 Boundingbox::intersect 一样的方法
        glm::vec3 inv_direction{ 1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z };
        float mx = (mesh.min_bound.x - ray.origin.x) * inv_direction.x;
        float Mx = (mesh.max_bound.x - ray.origin.x) * inv_direction.x;
        float my = (mesh.min_bound.y - ray.origin.y) * inv_direction.y;
        float My = (mesh.max_bound.y - ray.origin.y) * inv_direction.y;
        float mz = (mesh.min_bound.z - ray.origin.z) * inv_direction.z;
        float Mz = (mesh.max_bound.z - ray.origin.z) * inv_direction.z;
        float min = glm::max(glm::max(glm::min(mx, Mx), glm::min(my, My)), glm::min(mz, Mz));
        float Max = glm::min(glm::min(glm::max(mx, Mx), glm::max(my, My)), glm::max(mz, Mz));
        if (Max < 0 || min > Max)
            return -1.0f;
    }

    int min_idx = -1;
    float tmin = FLT_MAX;
    glm::vec3 barypos(0.0f), minbarypos(0.0f);
    for (int i = mesh.tri_start_idx; i < mesh.tri_end_idx; ++i) {
        auto& t = triangles[i];
        // 重心坐标 barypos:
        // The baryPosition output uses barycentric coordinates for the x and y components.The z component is the scalar factor for ray.
        // That is, 1.0 - baryPosition.x - baryPosition.y = actual z barycentric coordinate
        if (glm::intersectRayTriangle( 
            ray.origin, ray.direction, t.vertices[0], t.vertices[1], t.vertices[2], barypos
        )) {
            if (barypos.z > 0.0f && barypos.z < tmin) {
                min_idx = i;
                tmin = barypos.z;
                minbarypos = barypos;
            }
        }
    }
    if (min_idx == -1)
        return -1.0f;
    // 发生交叉，计算纹理坐标，用Triangle的vertices_texture_coord字段
    auto actual_z = 1.0f - minbarypos.x - minbarypos.y;
    auto& tri = triangles[min_idx];
    normal = actual_z * tri.vertex_normals[0] + tri.vertex_normals[1] + tri.vertex_normals[2];
    normal = glm::normalize(normal);
    texture_coord = actual_z * tri.vertices_texture_coord[0] + tri.vertices_texture_coord[1] + tri.vertices_texture_coord[2];
    return tmin;
}
// BVH树碰撞检测
__host__ __device__
bool BVHIntersectionTest(const Ray& r, int& hit_tri_id, ShadeableIntersection& isec,
    BVHTree* bvh_nodes, Triangle* triangles)
{
    if (!bvh_nodes)
        return false;
    bool hit = false;

    constexpr const int MAX_SEARCH_DEPTH = 128;
    int next_node_offset = 0, curr_idx = 0;
    int to_visit[MAX_SEARCH_DEPTH];
    bool dir_neg[3] = { r.direction.x < 0.0f,r.direction.y < 0.0f,r.direction.z < 0.0f };
    glm::vec3 inv_dir(1.0f / r.direction.x, 1.0f / r.direction.y, 1.0f / r.direction.z);

    while (true) {
        auto node = &bvh_nodes[curr_idx];
        float tmp_t = 0.0f;
        if (node->bounds.intersect(r, inv_dir)) {
            if (node->sub_areas > 0) {
                for (int i = 0; i < node->sub_areas; ++i) {
                    ShadeableIntersection inter;
                    if (triangles[node->first_area_idx + i].intersect(r, inter)) {
                        hit = true;
                        if ((isec.t == -1.0f) || (inter.t < isec.t)) {
                            isec = inter;
                            hit_tri_id = triangles[node->first_area_idx + i].id;
                        }
                    }
                }
                if (next_node_offset == 0)
                    break;
                curr_idx = to_visit[--next_node_offset];
            }
            else {
                if (next_node_offset == MAX_SEARCH_DEPTH) {
                    curr_idx = to_visit[--next_node_offset];
                    continue;
                }
                if (dir_neg[node->Axis]) {
                    to_visit[next_node_offset++] = curr_idx + 1;
                    curr_idx = node->RChild_idx;
                }
                else {
                    to_visit[next_node_offset++] = node->RChild_idx;
                    curr_idx = curr_idx + 1;
                }
            }
        }
        else {
            if (next_node_offset == 0)
                break;
            curr_idx = to_visit[--next_node_offset];
        }
    }

    return hit;
}