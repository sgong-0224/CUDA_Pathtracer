#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
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
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    Ray newray;
    newray.origin = intersect;

    // RGB -> luma
    const auto luma_vec = glm::vec3(0.2126, 0.7152, 0.0722);
    float diffuse_luma = glm::dot(m.color, luma_vec);
    float specular_luma = glm::dot(m.specular.color, luma_vec);
    float diffuse_rate = diffuse_luma / (diffuse_luma + specular_luma);

    thrust::uniform_real_distribution<float> u01(0, 1);
    if (u01(rng) < diffuse_rate) {
        newray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color / diffuse_rate;
    } else {
        newray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color / (1 - diffuse_rate);
    }
    pathSegment.ray = newray;
}
