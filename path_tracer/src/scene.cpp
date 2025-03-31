#include <iostream>
#include <cstring>
#include <omp.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
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
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading
        const std::vector<float> default_color = { 0.0f,0.0f,0.0f };
        const auto& col = p.value("RGB", default_color);
        const auto& spec_col = p.value("SPECRGB", default_color);
        newMaterial.specular.exponent = p.value("SPECEX",0.0f);
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
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "mesh") {   
            // 单独处理网格
            newGeom.type = MESH;
            string filename = (cfg_path.parent_path()/"Models"/p["MESH_FILE"]).string();
            newGeom.tri_start_idx = triangles.size();
            tinyobj::attrib_t attr;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string err;
            bool status = tinyobj::LoadObj(&attr, &shapes, &materials, &err, filename.c_str());
            if (!err.empty())
                std::cout << err << '\n';
            if (!status)
                exit(1);

            // 为图形加载三角形
            auto& norm = attr.normals;
            auto& texcoord = attr.texcoords;
            auto& vertices = attr.vertices;

            omp_set_nested(1);
#pragma omp parallel for
            for (size_t i = 0; i < shapes.size(); ++i) {
                size_t idx = 0;
                #pragma omp parallel for
                for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); ++j) {
                    size_t fv = shapes[i].mesh.num_face_vertices[j];
                    Triangle tri;
                    for (size_t v = 0; v < fv; ++v) {
                        tinyobj::index_t mesh_id = shapes[i].mesh.indices[idx + v];
                        
                        tinyobj::real_t vx = attr.vertices[3 * mesh_id.vertex_index + 0];
                        tinyobj::real_t vy = attr.vertices[3 * mesh_id.vertex_index + 1];
                        tinyobj::real_t vz = attr.vertices[3 * mesh_id.vertex_index + 2];
                        tri.vertices[v] = glm::vec3(vx, vy, vz);

                        if(mesh_id.normal_index >= 0){
                            tinyobj::real_t norm_x = attr.normals[3 * mesh_id.normal_index + 0];
                            tinyobj::real_t norm_y = attr.normals[3 * mesh_id.normal_index + 1];
                            tinyobj::real_t norm_z = attr.normals[3 * mesh_id.normal_index + 2];
                            tri.vertex_normals[v] = glm::vec3(norm_x, norm_y, norm_z);
                        }
                        
                        if (mesh_id.texcoord_index >= 0) {
                            tinyobj::real_t texcoord_u = attr.texcoords[2 * mesh_id.texcoord_index + 0];
                            tinyobj::real_t texcoord_v = attr.texcoords[2 * mesh_id.texcoord_index + 1];
                            tri.vertices_texture_coord[v] = glm::vec2(texcoord_u, texcoord_v);
                        }
                    }
                    idx += fv;
                    tri.surface_normal = glm::cross(tri.vertices[1] - tri.vertices[0], tri.vertices[2] - tri.vertices[0]);
                    triangles.emplace_back(tri);
                }
            }

            // 加载三角形
            newGeom.n_tris = triangles.size() - newGeom.tri_start_idx;
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
            // 计算BoundingBox
            newGeom.min_bound = triangles[0].vertices[0];
            newGeom.max_bound = triangles[0].vertices[0];
            for (auto& tri : triangles) {
                for (int i = 0; i < 3; ++i) {
                    // 对mesh, 将缩放应用到三角形上，全部完成后还原几何体本身的缩放
                    tri.vertices[i].x *= newGeom.scale[0];
                    tri.vertices[i].y *= newGeom.scale[1];
                    tri.vertices[i].z *= newGeom.scale[2];
                    newGeom.min_bound = glm::min(newGeom.min_bound, tri.vertices[i]);
                    newGeom.max_bound = glm::max(newGeom.max_bound, tri.vertices[i]);
                }
            }
            newGeom.scale = glm::vec3(1.0f);
            // TODO: scene boundingbox

        }
        else {
            if (type == "cube")
                newGeom.type = CUBE;
            else if (type == "sphere")
                newGeom.type = SPHERE;
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
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
}
