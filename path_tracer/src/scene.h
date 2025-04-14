#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

#include "tiny_obj_loader.h"
#include "boundingbox.h"
#include "BVH_tree.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(string filename);
    ~Scene();
    
    BVHTree* bvh_tree{ nullptr };
    int n_bvh_nodes = 0;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    std::vector<BoundingBox> bounding_boxes;
    std::vector<Texture> textures;
    RenderState state;
};
