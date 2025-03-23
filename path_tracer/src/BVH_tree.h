#ifndef BVH_TREE_H
#define BVH_TREE_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>
#include "sceneStructs.h"
#include "boundingbox.h"

class Triangle;

class BVH_BBox {
public:
	int index;
	BoundingBox bounds;
	glm::vec3 center;
	BVH_BBox(){}
	BVH_BBox(int i, const BoundingBox& b): 
		index(i), bounds(b.min_corner, b.max_corner), center(0.5f*(b.min_corner+b.max_corner)){}
};

class SurfaceAreaHeuristic {
public:
	int count = 0;
	BoundingBox bounds;
};

class BVHTreeNode {
public:
	BoundingBox  bbox{};
	BVHTreeNode* LChild{ nullptr };
	BVHTreeNode* RChild{ nullptr };
	int Axis;
	int sub_areas;
	int first_area_idx;

	BVHTreeNode() {}
	void mkLeaf(int first_offset, int n, const BoundingBox& b) {
		sub_areas = n;
		first_area_idx = first_offset;
		bbox = b;
		Axis = -1;
	}
	void mkNode(int axis, BVHTreeNode*l, BVHTreeNode* r) {
		LChild = l;
		RChild = r;
		Axis = axis;
		bbox = LChild->bbox || RChild->bbox;
		sub_areas = 0;
	}
};

class BVHTree {		// 以数组形式存储
public:
	BoundingBox bounds;
	int sub_areas;
	int Axis;
	int first_area_idx;
	int RChild_idx;
};

BVHTree* build_bvh_tree(int& n_nodes, std::vector<Triangle>& triangles); 
void delete_bvh(BVHTree* tree);
BVHTreeNode* build_bvh(std::vector<BVH_BBox>& bounding_boxes,std::vector<Triangle>& ordered_triangles,std::vector<Triangle>& triangles,
	int start, int end, int& n_nodes);
void delete_bvh_tree(BVHTreeNode* root);
int traverse_bvh(BVHTreeNode* node, BVHTree* tree, int& offset);

#endif