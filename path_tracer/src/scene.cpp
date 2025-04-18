#include <iostream>
#include <cstring>
#include <omp.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <map>
#include "json.hpp"
#include <filesystem>

#define TINYOBJLOADER_IMPLEMENTATION
#include "scene.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    auto cfg_path = std::filesystem::absolute(jsonName);
    
    // Material
    const auto& materialsData = data["Materials"];
    std::map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading
        const std::vector<float> default_color = { 0.0f,0.0f,0.0f };
        const auto& col = p.value("RGB", default_color);
        const auto& spec_col = p.value("SPECRGB", col);
        newMaterial.specular.exponent = p.value("SPECEX",1.0f);
        newMaterial.hasReflective = p.value("REFLECTIVE",0.0f);
        newMaterial.emittance = p.value("EMITTANCE",0.0f);
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        const auto& texture_path = p.value("TEXTURE_FILE", string());
        newMaterial.specular.color = glm::vec3(spec_col[0], spec_col[1], spec_col[2]);
        MatNameToID[name] = materials.size();
        // 处理自定义纹理
        newMaterial.texture_id = -1;
        
        if (!texture_path.empty()) {
            string filename = (cfg_path.parent_path() / "Textures" / texture_path).string();
            Texture texture;
            bool status = texture.load(filename.c_str());
            if (!status) {
                std::cout << "Texture load error!\n";
                exit(1);
            }
            newMaterial.texture_id = textures.size();
            textures.emplace_back(texture);
        }
        
        materials.emplace_back(newMaterial);
    }

    // Object/Geom
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        Geom newGeom;
        // 填充几何体信息
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        const auto& type = p["TYPE"];
        if (type == "mesh") {   
            // 单独处理网格
            newGeom.type = MESH;
            string obj_filename = (cfg_path.parent_path()/"Models"/p["OBJ_FILE"]).string();
            newGeom.tri_start_idx = triangles.size();
            newGeom.boundingbox_idx = bounding_boxes.size();
            tinyobj::attrib_t attr;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string err;
            bool status = tinyobj::LoadObj(&attr, &shapes, &materials, &err, obj_filename.c_str());
            if (!err.empty())
                std::cout << err << '\n';
            if (!status)
                exit(1);

            // 填充三角形信息
            auto& norm = attr.normals;
            auto& texcoord = attr.texcoords;
            auto& vertices = attr.vertices;
            float min_X = FLT_MAX, min_Y = FLT_MAX, min_Z = FLT_MAX;
            float max_X = FLT_MIN, max_Y = FLT_MIN, max_Z = FLT_MIN;
            for (size_t i = 0; i < shapes.size(); ++i) {
                size_t idx = 0;
                // 在加载过程中应用 transform，并计算 BoundingBox
                for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); ++j) {
                    size_t fv = shapes[i].mesh.num_face_vertices[j];
                    Triangle tri;
#pragma omp parallel for
                    for (size_t v = 0; v < fv; ++v) {
                        tinyobj::index_t mesh_id = shapes[i].mesh.indices[idx + v];
                        
                        tinyobj::real_t vx = attr.vertices[3 * mesh_id.vertex_index + 0];
                        tinyobj::real_t vy = attr.vertices[3 * mesh_id.vertex_index + 1];
                        tinyobj::real_t vz = attr.vertices[3 * mesh_id.vertex_index + 2];
                        tri.vertices[v] = glm::vec3(newGeom.transform * glm::vec4(vx, vy, vz, 1.0f));

                        if(mesh_id.normal_index >= 0){
                            tinyobj::real_t norm_x = attr.normals[3 * mesh_id.normal_index + 0];
                            tinyobj::real_t norm_y = attr.normals[3 * mesh_id.normal_index + 1];
                            tinyobj::real_t norm_z = attr.normals[3 * mesh_id.normal_index + 2];
                            tri.vertex_normals[v] = glm::vec3(newGeom.invTranspose * 
                                                    glm::vec4(norm_x, norm_y, norm_z, 0.0f));
                        }
                        
                        if (mesh_id.texcoord_index >= 0) {
                            tinyobj::real_t texcoord_u = attr.texcoords[2 * mesh_id.texcoord_index + 0];
                            tinyobj::real_t texcoord_v = attr.texcoords[2 * mesh_id.texcoord_index + 1];
                            tri.vertices_texture_coord[v] = glm::vec2(texcoord_u, texcoord_v);
                        }
                    }
                    glm::vec3 min_p, max_p;
                    tri.calculate_boundaries( min_p, max_p );
#pragma omp parallel num_threads(6)
                    {
#pragma omp sections nowait
                        {
#pragma omp section
                            min_X = min(min_p.x, min_X);
#pragma omp section
                            min_Y = min(min_p.y, min_Y);
#pragma omp section
                            min_Z = min(min_p.z, min_Z);
#pragma omp section
                            max_X = max(max_p.x, max_X);
#pragma omp section
                            max_Y = max(max_p.y, max_Y);
#pragma omp section
                            max_Z = max(max_p.z, max_Z);
                        }
                    }
                    idx += fv;
                    tri.id = triangles.size();
                    triangles.emplace_back(tri);
                }
            }
            newGeom.tri_end_idx = triangles.size();
            newGeom.min_bound = glm::vec3(min_X, min_Y, min_Z);
            newGeom.max_bound = glm::vec3(max_X, max_Y, max_Z);
            bounding_boxes.emplace_back(newGeom.min_bound, newGeom.max_bound);
        }
        else {
            if (type == "cube")
                newGeom.type = CUBE;
            else if (type == "sphere")
                newGeom.type = SPHERE;
        }
        geoms.emplace_back(newGeom);
    }

    // Camera
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
    // build BVH Tree
    bvh_tree = build_bvh_tree(n_bvh_nodes, triangles);
}
