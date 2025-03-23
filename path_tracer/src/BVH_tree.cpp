#include "BVH_tree.h"

constexpr int MAX_AREAS = 8;
int CMP_AXIS = 0;
bool compare_bbox(const BVH_BBox& a, const BVH_BBox& b)
{
	return a.center[CMP_AXIS] < b.center[CMP_AXIS];
}

void delete_bvh_tree(BVHTreeNode* root)
{
	if (!root->LChild && !root->RChild) {
		delete root;
		return;
	}
	delete_bvh_tree(root->LChild);
	delete_bvh_tree(root->RChild);
	delete root;
	return;
}
void delete_bvh(BVHTree* tree)
{
	delete[] tree;
	return;
}
// 用表面积启发式构建
BVHTreeNode* build_bvh(
	std::vector<BVH_BBox>& bounding_boxes, 
	std::vector<Triangle>& ordered_triangles,
	std::vector<Triangle>& triangles, 
	int start, int end, int& n_nodes)
{
	BVHTreeNode* newnode = new BVHTreeNode();
	n_nodes += 1;
	auto bounds = bounding_boxes[start].bounds;
#pragma omp parallel for
	for (int i = start; i < end; ++i)
		bounds = bounds || bounding_boxes[i].bounds;
	int n_tris = end - start;
	if (n_tris == 1) {
		int first_offset = ordered_triangles.size();
		for (int i = start; i < end; ++i)
			ordered_triangles.emplace_back(triangles[bounding_boxes[i].index]);
		newnode->mkLeaf(first_offset, n_tris, bounds);
		return newnode;
	}

	auto central_bound = BoundingBox(bounding_boxes[start].center, bounding_boxes[start].center);
#pragma omp parallel for
	for (int i = start; i < end; ++i)
		central_bound = central_bound || bounding_boxes[i].center;
	int longest_axis = central_bound.longest_axis();
	if (central_bound.min_corner[longest_axis] == central_bound.max_corner[longest_axis]) {
		// 长轴为0，叶节点
		int first_offset = ordered_triangles.size();
		for (int i = start; i < end; ++i)
			ordered_triangles.emplace_back(triangles[bounding_boxes[i].index]);
		newnode->mkLeaf(first_offset, n_tris, bounds);
		return newnode;
	}

	float mid;
	if (n_tris == 2) {
		mid = 1.0f * (start + end) / 2.0f;
		CMP_AXIS = longest_axis;
		std::nth_element(&bounding_boxes[start], &bounding_boxes[(int)mid], &bounding_boxes[end - 1] + 1, compare_bbox);
		newnode->mkNode(
			longest_axis,
			build_bvh(bounding_boxes, ordered_triangles, triangles, start, (int)mid, n_nodes),
			build_bvh(bounding_boxes, ordered_triangles, triangles, (int)mid, end, n_nodes)
		);
		return newnode;
	} else {
		// 表面积启发式划分
		constexpr int n_regions = MAX_AREAS - 1;
		SurfaceAreaHeuristic regions[n_regions];
#pragma omp parallel for
		for (int i = start; i < end; ++i) {
			int idx = n_regions * central_bound.getOffsetBoxes(bounding_boxes[i].center)[longest_axis];
			if (idx == n_regions)
				idx = n_regions - 1;
			regions[idx].count += 1;
			regions[idx].bounds = regions[idx].bounds || bounding_boxes[i].bounds;
		}

		float cost[n_regions - 1];
		for (int i = 0; i < n_regions - 1; ++i) {
			int count_0 = 0, count_1 = 0;
			BoundingBox Area0, Area1;
#pragma omp parallel for
			for (int j = 0; j < i; ++j) {
				count_0 += regions[j].count;
				Area0 = Area0 || regions[j].bounds;
			}
#pragma omp parallel for
			for (int j = i + 1; j < n_regions; ++j) {
				count_1 += regions[j].count;
				Area1 = Area1 || regions[j].bounds;
			}
			cost[i] = 1.0f * (count_0 * Area0.area() + count_1 * Area1.area()) / bounds.area();
		}

		float min_cost = FLT_MAX;
		int split_idx = 0;
		for (int i = 0; i < n_regions - 1; ++i) {
			if (cost[i] < min_cost) {
				min_cost = cost[i];
				split_idx = i;
			}
		}

		if( min_cost >= n_tris && n_tris <= MAX_AREAS ){
			// 启发式划分并非最优
			int first_offset = ordered_triangles.size();
			for (int i = start; i < end; ++i)
				ordered_triangles.emplace_back(triangles[bounding_boxes[i].index]);
			newnode->mkLeaf(first_offset, n_tris, bounds);
			return newnode;
		}

		BVH_BBox* mid_ptr = std::partition(&bounding_boxes[start], &bounding_boxes[end - 1] + 1,
			[=](const BVH_BBox& b) {
				int idx = n_regions * central_bound.getOffsetBoxes(b.center)[longest_axis];
				if (idx == n_regions)
					idx = n_regions - 1;
				return idx <= split_idx;
			});
		mid = mid_ptr - &bounding_boxes[0];
		newnode->mkNode(
			longest_axis,
			build_bvh(bounding_boxes, ordered_triangles, triangles, start, (int)mid, n_nodes),
			build_bvh(bounding_boxes, ordered_triangles, triangles, (int)mid, end, n_nodes)
		);
		return newnode;
	}
}
// 迭代DFS遍历，返回下一个节点的索引
int traverse_bvh(BVHTreeNode* node, BVHTree* tree, int& offset)
{
	BVHTree* t = &tree[offset];
	t->bounds = node->bbox;
	int next_offset = offset++;
	if (node->sub_areas > 0) {
		// 叶节点
		t->first_area_idx = node->first_area_idx;
		t->sub_areas = node->sub_areas;
		return next_offset;
	}
	t->sub_areas = 0;
	t->Axis = node->Axis;
	traverse_bvh(node->LChild, tree, offset);
	t->RChild_idx = traverse_bvh(node->RChild, tree, offset);
	return next_offset;
}
// 把BVH树展平，返回一个数组
BVHTree* build_bvh_tree(int& n_nodes, std::vector<Triangle>& triangles)
{
	if (triangles.size() == 0)
		return nullptr;
	auto n_tris = triangles.size();
	std::vector<BVH_BBox> bounding_boxes(n_tris);
	// 获取BoundingBox信息
#pragma omp parallel for
	for (int i = 0; i < n_nodes; ++i)
		bounding_boxes[i] = BVH_BBox(i, triangles[i].getBoundingBox());
	// 创建BVH树
	std::vector<Triangle> ordered_triangles;
	ordered_triangles.reserve(n_tris);
	n_nodes = 0;
	BVHTreeNode* root;
	root = build_bvh(bounding_boxes, ordered_triangles, triangles, 0, n_tris, n_nodes);
	// 展平
	BVHTree* tree = new BVHTree[n_nodes];
	int offset = 0;
	traverse_bvh(root, tree, offset);
	delete_bvh_tree(root);
	return nullptr;
}