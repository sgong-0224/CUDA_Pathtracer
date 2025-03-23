#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>

struct Ray;

class BoundingBox {
public:
	glm::vec3 min_corner;
	glm::vec3 max_corner;

	BoundingBox() : min_corner(glm::vec3(0.0f)), max_corner(glm::vec3(0.0f)) {}
	BoundingBox(const glm::vec3& min, const glm::vec3& max) : min_corner(min), max_corner(max) {}
	BoundingBox(const BoundingBox& bb) : min_corner(bb.min_corner), max_corner(bb.max_corner) {}

	int longest_axis() const
	{
		glm::vec3 dist = max_corner - min_corner;
		return (dist.x > dist.y && dist.x > dist.z) ? 0 :
			(dist.y > dist.z) ? 1 : 2;
	}
	bool is_point_inside(const glm::vec3& point) const
	{
		return (point.x >= min_corner.x && point.x <= max_corner.x) &&
			(point.y >= min_corner.y && point.y <= max_corner.y) &&
			(point.z >= min_corner.z && point.z <= max_corner.z);
	}
	float area() const
	{
		glm::vec3 d = max_corner - min_corner;
		return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
	}
	__host__ __device__
	BoundingBox& operator = (const BoundingBox& bbox)
	{
		min_corner = bbox.min_corner;
		max_corner = bbox.max_corner;
		return *this;
	}

	// 取两个BoundingBox的并集
	BoundingBox operator || (BoundingBox& bbox)
	{
		if (this->min_corner.x == 0.0f && this->min_corner.y == 0.0f && this->min_corner.z == 0.0f
			&& this->max_corner.x == 0.0f && this->max_corner.y == 0.0f && this->max_corner.z == 0.0f)
			return bbox;
		return BoundingBox{
			glm::vec3(
				glm::min(bbox.min_corner.x, this->min_corner.x),
				glm::min(bbox.min_corner.y, this->min_corner.y),
				glm::min(bbox.min_corner.z, this->min_corner.z)
			),
			glm::vec3(
				glm::max(bbox.max_corner.x, this->max_corner.x),
				glm::max(bbox.max_corner.y, this->max_corner.y),
				glm::max(bbox.max_corner.z, this->max_corner.z)
			)
		};
	}
	// 取覆盖某个点p的最小BoundingBox
	BoundingBox operator || (const glm::vec3& p)
	{
		return BoundingBox{
			glm::vec3(glm::min(p.x, this->min_corner.x), glm::min(p.y, this->min_corner.y), glm::min(p.z, this->min_corner.z)),
			glm::vec3(glm::max(p.x, this->max_corner.x), glm::max(p.y, this->max_corner.y), glm::max(p.z, this->max_corner.z))
		};
	}
	// 判断一条路径是否与BoundingBox相交
	__host__ __device__
	bool intersect(const Ray& ray, const glm::vec3& inv_direction)
	{
		// P(t) = O + tD, 确定参数t的范围
		float mx = (min_corner.x - ray.origin.x) * inv_direction.x;
		float Mx = (max_corner.x - ray.origin.x) * inv_direction.x;
		float my = (min_corner.y - ray.origin.y) * inv_direction.y;
		float My = (max_corner.y - ray.origin.y) * inv_direction.y;
		float mz = (min_corner.z - ray.origin.z) * inv_direction.z;
		float Mz = (max_corner.z - ray.origin.z) * inv_direction.z;

		float min = glm::max(glm::max(glm::min(mx, Mx), glm::min(my, My)), glm::min(mz, Mz));
		float Max = glm::min(glm::min(glm::max(mx, Mx), glm::max(my, My)), glm::max(mz, Mz));

		if (Max < 0)	// 反方向
			return false;
		if (min > Max)
			return false;
		return true;
	}
	// 计算到达点p偏移的BoundingBox数量
	glm::vec3 getOffsetBoxes(const glm::vec3& p) const
	{
		glm::vec3 offset = p - min_corner;
		if (max_corner.x > min_corner.x)
			offset.x /= (max_corner.x - min_corner.x);
		if (max_corner.y > min_corner.y)
			offset.y /= (max_corner.y - min_corner.y);
		if (max_corner.z > min_corner.z)
			offset.z /= (max_corner.z - min_corner.z);
		return offset;
	}
};

#endif