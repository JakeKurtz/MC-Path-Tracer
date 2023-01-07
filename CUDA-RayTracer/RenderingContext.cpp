#include "RenderingContext.h"

RenderingContext::RenderingContext()
{
    this->path_tracer = std::make_shared<PathTracer>();
    this->raster = std::make_shared<Rasterizer>();
    this->overlay = std::make_shared<Overlay>();
}

RenderingContext::RenderingContext(int width, int height)
{
    this->width = width;
    this->height = height;

    this->path_tracer = std::make_shared<PathTracer>();
    this->raster = std::make_shared<Rasterizer>(width, height);
    this->overlay = std::make_shared<Overlay>(width, height);

    window_flags |= ImGuiWindowFlags_MenuBar;
    window_flags |= ImGuiWindowFlags_NoCollapse;
}

void RenderingContext::frame_time()
{
    float current_frame_time = glfwGetTime();
    delta_time = current_frame_time - last_frame_time;
    last_frame_time = current_frame_time;
}

void RenderingContext::begin_imGUI()
{
    if (!ImGui::Begin(window_name.c_str(), NULL, window_flags))
    {
        // Early out if the window is collapsed, as an optimization.
        //ImGui::End();
        return;
    }
}

void RenderingContext::end_imGUI()
{
    ImGui::End();
}

void RenderingContext::init_image_buffer()
{
    image_buffer = new char[width * height * 3];

    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    glBindTexture(GL_TEXTURE_2D, 0);
}

void RenderingContext::update_window_properties()
{
    window_size = ImGui::GetWindowSize();

    window_min_pos = ImGui::GetWindowContentRegionMin();
    window_max_pos = ImGui::GetWindowContentRegionMax();

    window_min_pos.x += ImGui::GetWindowPos().x;
    window_min_pos.y += ImGui::GetWindowPos().y;
    window_max_pos.x += ImGui::GetWindowPos().x;
    window_max_pos.y += ImGui::GetWindowPos().y;

    window_center.x = (window_max_pos.x + window_min_pos.x) / 2.f;
    window_center.y = (window_max_pos.y + window_min_pos.y) / 2.f;
}

void RenderingContext::draw_frame_to_window()
{
    ImGuiIO& io = ImGui::GetIO();

    io.ConfigWindowsMoveFromTitleBarOnly = true;

    ImVec2 mp = ImGui::GetIO().MousePos;

    float ratio = max((float)window_size.x / (float)width, (float)window_size.y / (float)height); // scale image to window size
    float height_p = height * ratio;
    float width_p = width * ratio;

    ImVec2 cursor_pos = ImVec2((window_size.x - width_p) * 0.5, (window_size.y - height_p) * 0.5);
    ImGui::SetCursorPos(cursor_pos); // center image

    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddImage((void*)image_texture,
        pos,
        ImVec2(pos.x + (width_p), pos.y + (height_p)),
        ImVec2(0, 1),
        ImVec2(1, 0));
}

void RenderingContext::bind_fbo_to_frame(GLuint frame_buffer_texture)
{
    glBindTexture(GL_TEXTURE_2D, image_texture);
    glGetTextureImage(frame_buffer_texture, 0, GL_RGB, GL_UNSIGNED_BYTE, width * height * 3, image_buffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_buffer);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void RenderingContext::draw_menu_bar()
{
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("Image"))
        {
            if (ImGui::MenuItem("Save")) {
                stbi_flip_vertically_on_write(true);
                stbi_write_png("../scene_output.png", width, height, 3, image_buffer, width * 3);
            }
            ImGui::MenuItem("Save As");
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View"))
        {
            if (ImGui::BeginMenu("Overlay"))
            {

                ImGui::Checkbox("Enable Overlay", &enable_overlay);
                ImGui::Checkbox("Enable XRay", &overlay->enable_xray);

                if (ImGui::BeginMenu("Grid"))
                {
                    ImGui::Checkbox("Enable", &overlay->enable_grid);
                    ImGui::Checkbox("Flat", &overlay->grid_flat);

                    ImGui::Checkbox("Axis Highlight", &overlay->grid.highlight_axis);
                    ImGui::Checkbox("WRT Origin", (bool*)&overlay->grid.fade_mode);

                    ImGui::DragFloat("Grid Scale##grid_scale", &overlay->grid.scale, 0.01f, 0.f, 1000.f, "%.2f");
                    ImGui::DragFloat("Grid Fade Distance##grid_dist", &overlay->grid.fade_far, 0.01f, 0.f, 1000.f, "%.2f");
                    ImGui::DragFloat("min dist line width##min_dist_line_width", &overlay->grid.max_dist_line_width, 0.01f, 0.f, 1000.f, "%.2f");
                    ImGui::DragFloat("max dist_line width##max_dist_line_width", &overlay->grid.min_dist_line_width, 0.01f, 0.f, 1000.f, "%.2f");

                    ImGui::ColorEdit4("Color#color", (float*)&overlay->grid.color);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Wireframe"))
                {
                    ImGui::Checkbox("Enable", &overlay->enable_wireframe);
                    ImGui::ColorEdit4("Color#color", (float*)&overlay->wireframe.color);
                    ImGui::DragFloat("Thickness##thickness", &overlay->wireframe.thickness, 0.01f, 0.f, 1000.f, "%.2f");

                    ImGui::EndMenu();
                }

                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Shading"))
        {
            int render_mode_before = render_mode;
            ImGui::Combo("Render Type", &render_mode, "OpenGL Rasterizer\0MC Path Tracer\0Debug PT\0Wireframe\0");

            if ((int)render_mode != render_mode_before)
                buffer_reset = true;

            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }
}

void RenderingContext::update_context_interaction_state()
{
    bool title_bar_hovered = ImGui::IsItemHovered();

    if (ImGui::IsWindowFocused() && !title_bar_hovered) {
        currently_interacting = true;
    }
    else {
        currently_interacting = false;
    }
}

GLuint RenderingContext::render_scene(int render_mode)
{
    GLuint output_texture = 0;

    // update scene?
    //ds->update();

    switch (render_mode) {
    case RASTER_MODE:
        //output_texture = raster->draw_scene(s, raster_output_type, width, height);
        break;
    case PATHTRACE_MODE:
        //output_texture = path_tracer->draw_debug(s->get_dScene());
        //output_texture = path_tracer->render_image(s->get_dScene());
        //output_texture = path_tracer->render_image_preview(s->get_dScene());
        //output_texture = path_tracer->render_image_tile(s->get_dScene());
        //output_texture = raster->draw_texture(output_texture);

        break;
    }

    return output_texture;
}

void RenderingContext::raster_props()
{
    ImGui::Text("Rasterizer");

    if (ImGui::CollapsingHeader("Debug"))
    {
        ImGui::Combo("Deferred Buffers", &raster_output_type, "Shaded\0Position\0Normal\0Albedo\0Metallic Rough AO\0Emission\0Depth\0Shadows\0");
    }

    if (ImGui::CollapsingHeader("Shadows"))
    {

    }
}

void RenderingContext::pathtracing_props()
{
    ImGui::Text("Path Tracing");

    if (ImGui::CollapsingHeader("Sampling"))
    {
        ImGui::Combo("Integrator", &integrator_mode, "Path Tracing\0Branched Path Tracing\0");
        if (integrator_mode == 0) {

            int max_samples = path_tracer->get_samples();

            ImGui::DragInt("Samples", &max_samples, 1, 1, 5000);

            if (max_samples != path_tracer->get_samples()) {
                path_tracer->set_samples(max_samples);
            }

        }
        if (integrator_mode == 1) {
            ImGui::Text("AA");
            ImGui::Separator();
            ImGui::Text("Diffuse");
            ImGui::Separator();
            ImGui::Text("Specular");
            ImGui::Separator();
        }
    }

    if (ImGui::CollapsingHeader("Paths"))
    {
        ImGui::Text("Path Count");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Paths generated per tile.");

        ImGui::Separator();

        ImGui::Text("Max Bounces");
        //ImGui::DragFloat("##fd", &camera->focal_distance, 0.01f, 0.f, 1000.f, "%.2f");
    }

    if (ImGui::CollapsingHeader("Performance"))
    {
        if (ImGui::TreeNode("Tiles")) {

           // int tile_size = path_tracer->tile_size;

            //ImGui::InputInt("Tile Size", &tile_size, 1, 1024, ImGuiInputTextFlags_EnterReturnsTrue);

            //if (tile_size != path_tracer->tile_size) {
            //    path_tracer->set_tile_size(tile_size, tile_size);
                //path_tracer->reset_image();
           // }

            ImGui::Text("Order");
            ImGui::TreePop();
        }
        ImGui::Separator();
    }

    if (ImGui::Button("render")) {
        //path_tracer->reset_image();
    }
}

void RenderingContext::render_properties()
{
    if (render_mode == PATHTRACE_MODE) pathtracing_props();
    else raster_props();
}

void RenderingContext::set_width(int width)
{
    this->width = width;
}

void RenderingContext::set_height(int height)
{
    this->height = height;
}

bool RenderingContext::focused()
{
    //if (ImGui::IsWindowFocused()) {
     //   return true;
    //}
   //else {
    //    return false;
    //}
    return currently_interacting;
}

std::shared_ptr<Scene> RenderingContext::get_scene()
{
    return s;
}

//std::shared_ptr<dScene> RenderingContext::get_dscene()
//{
//    return ds;
//}

std::shared_ptr<Camera> RenderingContext::get_camera()
{
    return nullptr;//s->active_camera;
}

/*
Scene* RenderingContext::get_scene()
{
    return s;
}

Camera* RenderingContext::get_camera()
{
    return camera;
}
*/