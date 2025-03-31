#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng
){
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    ShadeableIntersection& intersection,
    const Material &m,
    Texture* textures,
    thrust::default_random_engine &rng
){
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    auto& normal = intersection.surfaceNormal;

    Ray& newray = pathSegment.ray;
    newray.origin = intersect + 0.0001f * normal;
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    pathSegment.color *= m.texture_id != -1 ? textures[m.texture_id].get_color(intersection.texture_coord) : m.color;

    if (m.hasRefractive) {
        // ��Schlick���Ƽ�������
        float n = m.indexOfRefraction;
        float R_0 = ((n - 1) * (n - 1)) / ((n + 1) * (n + 1));
        float X = 1 - glm::abs(glm::dot(pathSegment.ray.direction, normal));
        float X_squared = X * X;
        float R = R_0 + (1 - R_0) * (X * X_squared * X_squared);
        if (R < u01(rng)) { // ����
            newray.direction = glm::refract(pathSegment.ray.direction, normal, n);
            normal = -normal;
            pathSegment.color *= m.color;
        } else { // ����
            newray.direction = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.color *= m.specular.color;
        }
    } else if(u01(rng)<m.hasReflective) { // ���淴��
        newray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color;
    } else { // ������
        newray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    }
    pathSegment.color *= m.color;
    glm::clamp(pathSegment.color, glm::vec3(0.0f), glm::vec3(1.0f));
}
