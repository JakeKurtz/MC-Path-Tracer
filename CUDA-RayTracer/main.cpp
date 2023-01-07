//#define PXL_FULLSCREEN

#include "GLCommon.h"
#include "globals.h"
#include <cstdlib>

#include "Scene.h"
#include "PathTracer.h"
#include "Rasterizer.h"
#include "PerspectiveCamera.h"

#include "../imgui_docking/imgui.h"
#include "../imgui_docking/misc/cpp/imgui_stdlib.h"
#include "../imgui_docking/backends/imgui_impl_glfw.h"
#include "../imgui_docking/backends/imgui_impl_opengl3.h"

#include <stb_image.h>
#include <stb_image_write.h>
#include "SceneViewWindow.h"
#include "ObjectEditWindow.h"
#include "MaterialPreviewWindow.h"
#include "LSystem.h"
#include "Turtle.h"
#include "SkeletonMesh.h"
#include "Quaternion.h"

const char* glsl_version = "#version 330 core";

int width = 1280;
int height = 720;

int render_mode = 0;

bool buffer_reset = false;

float lastX;
float lastY;
bool firstMouse = true;
bool mouseDown = false;
bool mouse_button_3_down = false;
bool click = false;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

static bool show_demo_window = true;
static bool show_another_window = false;
static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
glm::vec3 background_color = glm::vec3(0.0f);
GLuint image_texture;
char* image_buffer = new char[width * height * 3];

std::shared_ptr<Rasterizer> r;
std::shared_ptr<PathTracer> pt;

std::shared_ptr<Scene> s;
std::shared_ptr<dScene> ds;

int light_select_type = 0;
int environment_color_mode = 0;

std::shared_ptr<Film> active_film;
std::shared_ptr<Camera> active_camera;
std::shared_ptr<Scene> active_scene;
//std::shared_ptr<PerspectiveCamera> camera;// = std::make_shared<PerspectiveCamera>(glm::vec3(0.289340049, 4.11911869, 10.5660067), (float)width / (float)height, glm::radians(45.f), 0.01f, 10000.f);

static void glfw_error_callback(int error, const char* description);
static void glfw_window_size_callback(GLFWwindow* window, int width, int height);
static void glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos);
static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
static void glfw_mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
static void glfw_process_input(GLFWwindow* window);

static void glfw_init(GLFWwindow** window, const int width, const int height)
{
    //
    // INITIALIZE GLFW/GLAD
    //

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glsl_version = "#version 330";

    glfwWindowHint(GLFW_DEPTH_BITS, 0);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);
    //glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, 1);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef PXL_FULLSCREEN
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    *window = glfwCreateWindow(mode->width, mode->height, "GLFW / CUDA Interop", monitor, NULL);
#else
    * window = glfwCreateWindow(width, height, "GLFW / CUDA Interop", NULL, NULL);
#endif

    if (*window == NULL)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(*window);

    //if (glewInit() != GLEW_OK)
    //    exit(EXIT_FAILURE);

    glfwSetErrorCallback(glfw_error_callback);
    glfwSetKeyCallback(*window, glfw_key_callback);
    glfwSetFramebufferSizeCallback(*window, glfw_window_size_callback);
    glfwSetCursorPosCallback(*window, glfw_mouse_callback);
    glfwSetMouseButtonCallback(*window, glfw_mouse_button_callback);
    glfwSetScrollCallback(*window, glfw_mouse_scroll_callback);

    // set up GLAD
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // tell GLFW to capture our mouse
    glfwSetInputMode(*window, GLFW_CURSOR, GLFW_CURSOR);

    // ignore vsync for now
    glfwSwapInterval(0);

    // only copy r/g/b
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    GLFW_INIT = true;
}
static void imgui_init(GLFWwindow** window) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(*window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    show_demo_window = true;
    show_another_window = false;
    ImVec4 clear_color;

    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

}

float raw_mouse_scroll = 8.6153846153846f;
float image_scale = 1.f;
float image_pan_x = 0;
float image_pan_y = 0;

bool interact_with_scene_view = true;
bool interact_with_render_view = true;

bool show_scene_view = false;
bool show_render_view = false;

bool show_properties_view = false;

vector<RenderWindow*> contexts;
//vector<RenderingContext*> contexts;
RenderWindow* active_context;
//RenderingContext* active_context;

shared_ptr<Material> material_selected = nullptr;
shared_ptr<Curve> curve_selected = nullptr;
shared_ptr<LSystem> lsys_selected = nullptr;
shared_ptr<Mesh> mesh_selected = nullptr;
shared_ptr<EnvironmentLight> environment_light_selected = nullptr;
shared_ptr<DirectionalLight> dir_light_selected = nullptr;
shared_ptr<PointLight> point_light_selected = nullptr;

ObjectEditWindow* OEW;
MaterialPreviewWindow* MPW;

void curve_view() {

}

void camera_props()
{
    ImGui::Text("Camera");

    if (active_context != nullptr) {
        if (ImGui::CollapsingHeader("Lens"))
        {
            ImGui::Text("Type");

            ImGui::Text("Focal Length");
            //ImGui::DragFloat("##fd", &camera->focal_distance, 0.01f, 0.f, 1000.f, "%.2f");

            ImGui::Separator();
        }

        if (ImGui::CollapsingHeader("Depth of Field"))
        {

            float focal_distance = active_camera->get_focal_distance();
            ImGui::DragFloat("Focal Distance", &focal_distance, 0.01f, 0.f, 1000.f, "%.2f");

            if (ImGui::IsItemActive() && focal_distance != active_camera->get_focal_distance()) {
                active_camera->set_focal_distance(focal_distance);
            }

            if (ImGui::TreeNode("Apature")) {

                float lens_radius = active_camera->get_lens_radius();
                ImGui::DragFloat("Size", &lens_radius, 0.01f, 0.001f, 10.f, "%.4f");

                if (ImGui::IsItemActive() && lens_radius != active_camera->get_lens_radius()) {
                    active_camera->set_lens_radius(lens_radius);
                }
                ImGui::TreePop();
            }
            ImGui::Separator();
        }

        if (ImGui::CollapsingHeader("Film"))
        {
            float exposure = active_film->get_exposure();
            ImGui::DragFloat("Exposure", &exposure, 0.01f, 0.1f, 15.f, "%.2f");

            if (ImGui::IsItemActive() && exposure != active_film->get_exposure()) {
                active_film->set_exposure(exposure);
            }
            
            ImGui::Separator();
        }
    }
}

void curve_props() {
    ImGui::Text("Curve");

    if (curve_selected != nullptr) {
        ImGui::Text(("name: " + curve_selected->get_name()).c_str());

        int res = curve_selected->get_depth();
        ImGui::SliderInt("Resolution##resolution", (int*)&res, 0, 20);

        if (res != curve_selected->get_depth()) {
            curve_selected->set_depth(res);
            curve_selected->update();
        }

    }
}

void lsys_props() {

    ImGui::Begin("L-System");
    //ImGui::Text("L-System");

    if (lsys_selected != nullptr) {
        ImGuiInputTextFlags input_text_flags = ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackHistory;

        std::string axiom = lsys_selected->get_axiom();

        ImGui::Text("Premise"); ImGui::SameLine();
        bool enter = ImGui::InputText("##axiom", &axiom, input_text_flags);

        if (strcmp(axiom.c_str(), lsys_selected->get_axiom().c_str()) && enter) {
                lsys_selected->set_axiom(axiom);
                lsys_selected->build();
        }

        static std::string new_rule_str = "";
        static bool valid_rule = true;

        ImGui::Text("Add Rule"); ImGui::SameLine();
        if (!valid_rule) ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
        ImGui::InputText("##newrule", &new_rule_str);
        if (!valid_rule) ImGui::PopStyleColor();

        ImGui::SameLine();
        if (ImGui::Button("+")) {
            valid_rule = lsys_selected->add_rule(new_rule_str);
            if (valid_rule) {
                new_rule_str = "";
            }
        }

        int i = 0;
        for (auto &rule_set : lsys_selected->get_rules()) {
            int j = 0;
            for (auto rule : rule_set.second) {

                int rule_id = i + j;
                int sub_rule_id = j;

                std::string s = rule->get_input_str();

                std::string str_id = "##rule" + to_string(rule_id);
                std::string title ="Rule "+ to_string(rule_id);
                ImGui::Text(title.c_str()); ImGui::SameLine();
                bool enter = ImGui::InputText(str_id.c_str(), &s, input_text_flags);

                ImGui::SameLine();
                if (ImGui::Button(("x" + str_id).c_str())) {
                    lsys_selected->delete_rule(rule_set.first, sub_rule_id);
                }

                if (strcmp(s.c_str(), rule->get_input_str().c_str()) && enter) {
                    lsys_selected->delete_rule(rule_set.first, sub_rule_id);
                    lsys_selected->add_rule(s);
                    lsys_selected->build();
                }

                j++;
            }
            i++;
        }

        int gen = lsys_selected->get_generations();
        ImGui::SliderInt("Generations##gen", (int*)&gen, 0, 20);

        if (gen != lsys_selected->get_generations()) {
            lsys_selected->set_generations(gen);
            lsys_selected->build();
        }

        auto turtle = lsys_selected->get_turtle();

        float angle = turtle->default_angle;
        ImGui::DragFloat("Default Angle##angle", (float*)&angle, 0.1, 0, 360);

        if (angle != turtle->default_angle) {
            turtle->default_angle = angle;
            lsys_selected->build();
        }

    }
    ImGui::End();
}

void shadow_props() {
    ImGui::Text("Shadows");

    if (active_context != nullptr) {
        //ImGui::DragFloat("look at point offset##look_at_pos", (float*)&active_context->raster->lookat_point_offset, 0.1, 0, 1000);
        //ImGui::DragFloat("light size##light_size", (float*)&active_context->raster->lightSize, 0.1, 0, 1000);
        //ImGui::DragFloat("search area size##light_size", (float*)&active_context->raster->searchAreaSize, 0.1, 0, 100);
        //ImGui::DragFloat("shadow scaler##shadow_scaler", (float*)&active_context->raster->shadowScaler, 0.01, 0, 10);
        //ImGui::DragInt("kernel size##kernel size", (int*)&active_context->raster->kernel_size, 1, 1, 10);
    }
}

void scene_graph() 
{
    static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth;
    
    ImGui::Begin("Scene Graph");

    if (ImGui::TreeNode("SCENE"))
    {

        static int selection_mask = (1 << 2);
        int node_clicked = -1;
        int id = 0;

        if (ImGui::TreeNode("MODELS")) {
            for (auto const& mesh_pair : s->render_objects) {
                auto mesh = mesh_pair.second;

                ImGuiTreeNodeFlags node_flags = base_flags;
                const bool is_selected = (selection_mask & (1 << id)) != 0;
                if (is_selected) {
                    node_flags |= ImGuiTreeNodeFlags_Selected;
                }

                id = mesh->get_id();
                bool mesh_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, node_flags, ("MODEL: " + mesh->get_name()).c_str(), id);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    node_clicked = id;
                    //mesh_selected = mesh;
                    OEW->set_obj(mesh);
                }

                if (mesh_node_open) {

                    ImGuiTreeNodeFlags leaf_flags = node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                    ImGui::TreeNodeEx((void*)(intptr_t)id, leaf_flags, ("MATERIAL: " + mesh->get_material()->get_name()).c_str(), id);
                    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                        node_clicked = id;
                        material_selected = mesh->get_material();
                    }
                    ImGui::TreePop();
                }

                /*if (model_node_open)
                {
                    for (Mesh* mesh : mesh->get_meshes()) {

                        id = mesh->get_id();
                        bool mesh_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, node_flags, ("MESH: " + mesh->get_name()).c_str(), id);
                        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                            node_clicked = id;
                        }

                        if (mesh_node_open) {
                            
                            ImGuiTreeNodeFlags leaf_flags = node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                            ImGui::TreeNodeEx((void*)(intptr_t)id, leaf_flags, ("MATERIAL: " + mesh->get_material()->name).c_str(), id);
                            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                                node_clicked = id;
                                material_selected = mesh->get_material();
                            }
                            ImGui::TreePop();
                        }
                    }
                    ImGui::TreePop();
                }*/
            }

            if (node_clicked != -1)
            {
                selection_mask = (1 << node_clicked);
            }
            ImGui::TreePop();
        }
        //ImGui::TreePop();
        /*
        if (ImGui::TreeNode("CURVES")) {
            for (std::shared_ptr<RenderObject> r_obj : s->curves) {

                std::shared_ptr<Curve> curve = std::dynamic_pointer_cast<Curve>(r_obj);

                ImGuiTreeNodeFlags node_flags = base_flags;
                const bool is_selected = (selection_mask & (1 << id)) != 0;
                if (is_selected) {
                    node_flags |= ImGuiTreeNodeFlags_Selected;
                }

                id = curve->get_id();
                //bool model_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, node_flags, ("CURVE: " + curve->get_name()).c_str(), id);
                ImGuiTreeNodeFlags leaf_flags = node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                ImGui::TreeNodeEx((void*)(intptr_t)id, leaf_flags, ("CURVE: " + curve->get_name()).c_str(), id);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    node_clicked = id;
                    curve_selected = curve;
                    //OEW->set_curve(curve);
                }
                ImGui::TreePop();
            }
            
            if (node_clicked != -1)
            {
                selection_mask = (1 << node_clicked);
            }
            ImGui::TreePop();
        }
        */
        
        if (ImGui::TreeNode("LIGHTS")) {

            ImGuiTreeNodeFlags node_flags = base_flags;

            for (std::shared_ptr<DirectionalLight> light : s->dir_lights) {
                ImGuiTreeNodeFlags leaf_flags = node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                const bool is_selected = (selection_mask & (1 << id)) != 0;
                if (is_selected) {
                    node_flags |= ImGuiTreeNodeFlags_Selected;
                }

                id = light->get_id();
                bool model_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, leaf_flags, (light->get_name()).c_str(), id);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    node_clicked = id;
                    light_select_type = 1;
                    dir_light_selected = light;
                }
            }
            for (std::shared_ptr<PointLight> light : s->point_lights) {
                ImGuiTreeNodeFlags leaf_flags = node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                const bool is_selected = (selection_mask & (1 << id)) != 0;
                if (is_selected) {
                    node_flags |= ImGuiTreeNodeFlags_Selected;
                }

                id = light->get_id();
                bool model_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, leaf_flags, (light->get_name()).c_str(), id);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    node_clicked = id;
                    light_select_type = 2;
                    point_light_selected = light;
                }
            }
            if (s->environment_light != nullptr) {
                ImGuiTreeNodeFlags leaf_flags = node_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                const bool is_selected = (selection_mask & (1 << id)) != 0;
                if (is_selected) {
                    node_flags |= ImGuiTreeNodeFlags_Selected;
                }

                id = 0;
                bool model_node_open = ImGui::TreeNodeEx((void*)(intptr_t)id, leaf_flags, ("ENV: " + s->environment_light->get_texture_filepath()).c_str(), id);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    node_clicked = id;
                    light_select_type = 0;
                    environment_light_selected = s->environment_light;
                }
            }
            ImGui::TreePop();
        }
        ImGui::TreePop();
    }
    
    ImGui::End();
}

void material_props() 
{
    ImGui::Text("Material");

    if (material_selected != nullptr) {

        if (MPW->get_material()->get_id() != material_selected->get_id()) {
            MPW->set_material(material_selected);
        }
        
        ImGui::Text(("name: "+material_selected->get_name()).c_str());

        glm::vec3 mat_color = material_selected->get_base_color_factor();
        ImGui::ColorEdit3("Base Color##base_color", (float*)&mat_color, ImGuiColorEditFlags_NoInputs);

        if (ImGui::IsItemActive() && mat_color != material_selected->get_base_color_factor()) {
            material_selected->set_base_color_factor(mat_color);
        }

        ImGui::ColorEdit3("Emissive Color##emissive_color", (float*)&material_selected->get_emissive_color_factor(), ImGuiColorEditFlags_NoInputs);

        float mat_roughness = material_selected->get_roughness_factor();
        ImGui::SliderFloat("Roughness Factor", &mat_roughness, 0.01f, 1.0f);

        if (ImGui::IsItemActive() && mat_roughness != material_selected->get_roughness_factor()) {
            material_selected->set_roughness_factor(mat_roughness);
        }

        float mat_metal = material_selected->get_metallic_factor();
        ImGui::SliderFloat("Metallic Factor", &mat_metal, 0.01f, 1.0f);

        if (ImGui::IsItemActive() && mat_metal != material_selected->get_metallic_factor()) {
            material_selected->set_metallic_factor(mat_metal);
        }

        //glm::vec3 mat_fres = material_selected->get_fresnel();
        //ImGui::ColorEdit3("Reflection Factor", (float*)&mat_fres, ImGuiColorEditFlags_NoInputs);

        //if (ImGui::IsItemActive() && mat_fres != material_selected->get_fresnel()) {
            //material_selected->set_fresnel(mat_fres);
        //}
    }
}

void model_props() 
{
    ImGui::Text("Model");
}

void mesh_props() 
{
    ImGui::Text("Mesh");
}

void light_props() 
{
    ImGui::Text("Light");
    if (active_context != nullptr) {
        auto environment_light = active_context->get_scene()->get_environment_light();

        switch (light_select_type) {
        case 0:

            ImGui::Text("Environment");

            environment_color_mode = environment_light->get_light_type();

            ImGui::Combo("Color", &environment_color_mode, "RGB\0HRD Texture\0");

            if (environment_color_mode == 0) 
            {
                if (environment_color_mode != environment_light->get_light_type())
                {
                    environment_light->set_type((EnvironmentLightType)0);
                }

                auto color = environment_light->get_color();
                ImGui::ColorEdit3("Color##env_color", (float*)&color, ImGuiColorEditFlags_NoInputs);

                if (environment_light->get_light_type() != EnvironmentLightType::Color) {
                    environment_light->set_color(color);
                }

                if (ImGui::IsItemActive() && (color != environment_light->get_color())) {
                    environment_light->set_color(color);
                }
            }
            else if (environment_color_mode == 1) 
            {
            if (environment_color_mode != environment_light->get_light_type())
                {
                    environment_light->set_type((EnvironmentLightType)1);
                }
                static char buf1[1024] = "../hrdi/HDR_029_Sky_Cloudy_Env.hdr";

                ImGui::InputText("filepath", buf1, 1024);

                if (environment_light->get_light_type() != EnvironmentLightType::HRDI) {
                    environment_light->set_texture_filepath(buf1);
                }

                if (ImGui::Button("load")) {
                    // TODO: better error checking for null strings.
                    if (strcmp(buf1, environment_light->get_texture_filepath().c_str())) {
                        environment_light->set_texture_filepath(buf1);
                    }
                }
            }
            /*
            float ls = environment_light->get_ls();
            ImGui::DragFloat("Intensity##fd", &ls, 0.01f, 0.f, 1000.f, "%.2f");

            if (ImGui::IsItemActive() && ls != environment_light->get_ls()) {
                environment_light->set_ls(ls);
            }
            */
            break;
        case 1:
        {
            ImGui::Text("Directional");

            glm::vec3 dir = dir_light_selected->get_dir();
            ImGui::DragFloat3("Direction", (float*)&dir, 0.01, 0.f, 1.f);

            if (ImGui::IsItemActive() && dir != dir_light_selected->get_dir()) {
                dir_light_selected->set_dir(dir);
            }

            glm::vec3 color = dir_light_selected->get_color();
            ImGui::ColorEdit3("Color##dir_color", (float*)&color, ImGuiColorEditFlags_NoInputs);

            if (ImGui::IsItemActive() && color != dir_light_selected->get_color()) {
                dir_light_selected->set_color(color);
            }

            float intensity = dir_light_selected->get_ls();
            ImGui::DragFloat("Intensity##fd", &intensity, 0.01f, 0.f, 1000.f, "%.2f");

            if (ImGui::IsItemActive() && intensity != dir_light_selected->get_ls()) {
                dir_light_selected->set_ls(intensity);
            }
        }
        break;
        case 2:
        {
            ImGui::Text("Point");

            glm::vec3 pos = point_light_selected->getPosition();
            ImGui::DragFloat3("Position", (float*)&pos, 0.1, 0.f, 100.f);
            point_light_selected->setPosition(pos);

            if (ImGui::IsItemActive()) {

            }

            glm::vec3 color = point_light_selected->get_color();
            ImGui::ColorEdit3("Color##pnt_color", (float*)&color, ImGuiColorEditFlags_NoInputs);
            point_light_selected->set_color(color);

            if (ImGui::IsItemActive()) {

            }

            float intensity = point_light_selected->get_ls();
            ImGui::DragFloat("Intensity##fd", &intensity, 0.01f, 0.f, 1000.f, "%.2f");
            point_light_selected->set_ls(intensity);

            if (ImGui::IsItemActive()) {

            }
        }
        break;
        }
    }
}

void properties_view()
{
    if (!ImGui::Begin("Properties"))
    {
        // Early out if the window is collapsed, as an optimization.
        ImGui::End();
        return;
    }

    ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
    if (ImGui::BeginTabBar("MyTabBar", tab_bar_flags))
    {
        if (ImGui::BeginTabItem("Camera"))
        {
            camera_props();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Light"))
        {
            light_props();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Model"))
        {
            model_props();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Curve"))
        {
            curve_props();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Material"))
        {
            material_props();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Rendering"))
        {
            //if (active_context != nullptr) {
                //active_context->render_properties();
            //}
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void render_gui(GLFWwindow* window)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::DockSpaceOverViewport();

    properties_view();

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("import"))
            {
                //Do something
            }
            if (ImGui::MenuItem("save as"))
            {
                //Do something
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Scene"))
        {
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View"))
        {
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
    
    //curve_view();

    scene_graph();

    //lsys_props();

    //shadow_props();

    for (auto c : contexts) {
        if (c->focused()) {
            active_context = c;
            active_scene = c->get_scene();
            active_camera = c->get_camera();
            active_film = c->get_film();
        }

        c->draw();
    }

    //camera_props();
    // Rendering
    ImGui::Render();

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Update and Render additional Platform Windows
    // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
    //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
    ImGuiIO io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
}

int main(int argc, char** argv)
{
    int x;

    cudaGetDeviceCount(&x);
    cudaSetDevice(0);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    std::cout << "\n\ndevices:" << x << "\n\n" << std::endl;
    std::cout << "\nproperties:" << props.name << std::endl;

    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    cout << "using " << properties.multiProcessorCount << " multiprocessors" << endl;
    cout << "max threads per processor: " << properties.maxThreadsPerMultiProcessor << endl;

    cudaDeviceReset();

    cudaDeviceSetLimit(cudaLimitStackSize, 105536);

    GLFWwindow* window;
    glfw_init(&window, width, height);
    imgui_init(&window);

    //active_camera = camera;

    //color_environmentLight = std::make_shared<EnvironmentLight>(glm::vec3(0.f));
    //hrd_environmentLight = std::make_shared<EnvironmentLight>("../hrdi/HDR_029_Sky_Cloudy_Env.hdr");

    std::shared_ptr<DirectionalLight> light_1 = std::make_shared<DirectionalLight>();
    light_1->set_ls(100.f);
    light_1->set_dir(glm::vec3(0.262, 0.965, 0.f));
    light_1->set_color(glm::vec3(1));

    std::shared_ptr<DirectionalLight> light_2 = std::make_shared<DirectionalLight>();
    light_2->set_ls(0.f);
    light_2->set_dir(glm::vec3(-10));
    light_2->set_color(glm::vec3(1.f,0,0));

    std::shared_ptr<DirectionalLight> light_3 = std::make_shared<DirectionalLight>();
    light_3->set_ls(50.f);
    light_3->set_dir(glm::vec3(0.f, 0.98, 0.001));
    light_3->set_color(glm::vec3(0,1,0));

    s = std::make_shared<Scene>();
    s->set_environment_light(std::make_shared<EnvironmentLight>("../hrdi/night_free_Env.hdr"));

    //s->add_light(light_1);
    //s->add_light(light_2);
    //s->add_light(light_3);

    //auto s2 = std::make_shared<Scene>();
    //s->load("../models/sphere.glb");
    //s->load("../models/plane_cube_test.glb");
    //s->load("../models/Suzanne.glb");
    //s2->load("../models/Suzanne.glb");
    //s->load("../models/Cube.glb");
    //s->load("../models/Cube.glb");
    s->load("../models/scene_show_off_dragon.glb");
    //s->load("../models/greek_sculpture.glb");
    //s->load("../models/photograph.glb");
    //s->load("../models/scene_show_off_spheres.glb");

    std::vector<glm::vec3> control_points_test;
    control_points_test.push_back(glm::vec3(-1, 0, 0));
    control_points_test.push_back(glm::vec3(-1, 1, 0));
    control_points_test.push_back(glm::vec3(1, 1, 0));
    control_points_test.push_back(glm::vec3(1, 0, 0));

    std::shared_ptr<Bezier> curve_test = std::make_shared<Bezier>(control_points_test);

    //s->add_render_object(curve_test);

    std::shared_ptr<LSystem> L = std::make_shared<LSystem>();

    L->set_axiom("FFFA");
    L->add_rule("A -> !(0.6) \"(0.6)[&FFFA] //// [&FFFA] //// [&FFFA]");

    if (L->build()) {
        //s->add_render_object(L->get_turtle()->get_curve());
        //s->add_render_object(L->get_turtle()->get_mesh());
    }

   //s->add_render_object(curve_test);

    std::shared_ptr<RenderEngine> re = std::make_shared<RenderEngine>();

    SceneViewWindow* SVW = new SceneViewWindow(s, re);
    MPW = new MaterialPreviewWindow(256, 256, re);
    OEW = new ObjectEditWindow(re);

    lsys_selected = L;

    contexts.push_back(SVW);
    contexts.push_back(OEW);
    contexts.push_back(MPW);

    double lastTime = glfwGetTime();
    int nbFrames = 0;

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Per-frame time logic
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glfw_process_input(window);

        render_gui(window);

        glfwSwapBuffers(window);
        glfwPollEvents();
        
        /*
        {
            // Measure speed
            double currentTime = glfwGetTime();
            nbFrames++;
            if (currentTime - lastTime >= 1.0) { // If last prinf() was more than 1 sec ago
                // printf and reset timer
                printf("%f ms/frame\n", 1000.0 / double(nbFrames));
                nbFrames = 0;
                lastTime += 1.0;
            }
        }
        */
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}

static void glfw_error_callback(int error, const char* description)
{
    fputs(description, stderr);
}
static void glfw_window_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
    buffer_reset = true;
}
static void glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (active_context != nullptr) active_context->mouse_callback(window, xpos, ypos);
}
static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (active_context != nullptr) active_context->mouse_button_callback(window, button, action, mods);
}
static void glfw_mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (active_context != nullptr) active_context->mouse_scroll_callback(window, xoffset, yoffset);
}
static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{

}
static void glfw_process_input(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (active_context != nullptr) active_context->process_input(window);
}