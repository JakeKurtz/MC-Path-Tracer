#include "RenderWindow.h"

RenderWindow::RenderWindow(std::shared_ptr<RenderEngine> render_engine)
{
    (this)->render_engine = render_engine;

    window_flags |= ImGuiWindowFlags_MenuBar;
    window_flags |= ImGuiWindowFlags_NoCollapse;
}

void RenderWindow::draw()
{
    frame_time();

    ImGui::Begin(name.c_str(), &p_open, window_flags);

    auto _window_size_x = window_size.x;
    auto _window_size_y = window_size.y;

    update_context_interaction_state();
    update_window_properties();

    if (_window_size_x != window_size.x || _window_size_y != window_size.y)
    {
        film->set_size(window_size.x, window_size.y);
        camera->set_aspect_ratio((float)window_size.x/(float)window_size.y);
    }

    menu_bar();

    render();
    
    draw_frame_to_window();

    ImGui::End();
}

void RenderWindow::draw_frame_to_window()
{
    ImGuiIO& io = ImGui::GetIO();

    io.ConfigWindowsMoveFromTitleBarOnly = true;

    ImVec2 mp = ImGui::GetIO().MousePos;

    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    uint32_t width, height;
    film->get_size(width, height);
    drawList->AddImage((void*)film->get_image_tex(),
        pos,
        ImVec2(pos.x + width, pos.y + height),
        ImVec2(0, 1),
        ImVec2(1, 0));
}

void RenderWindow::menu_bar()
{
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("Image"))
        {
            if (ImGui::MenuItem("Save")) {
                //stbi_flip_vertically_on_write(true);
                //stbi_write_png("../scene_output.png", width, height, 3, image_buffer, width * 3);
            }
            ImGui::MenuItem("Save As");
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View"))
        {
            if (ImGui::BeginMenu("Overlay"))
            {

                ImGui::Checkbox("Enable Overlay", &enable_overlay);
                ImGui::Checkbox("Enable XRay", &render_engine->overlay->enable_xray);

                if (ImGui::BeginMenu("Grid"))
                {
                    ImGui::Checkbox("Enable", &render_engine->overlay->enable_grid);
                    ImGui::Checkbox("Flat", &render_engine->overlay->grid_flat);

                    ImGui::Checkbox("Axis Highlight", &render_engine->overlay->grid.highlight_axis);
                    ImGui::Checkbox("WRT Origin", (bool*)&render_engine->overlay->grid.fade_mode);

                    ImGui::DragFloat("Grid Scale##grid_scale", &render_engine->overlay->grid.scale, 0.01f, 0.f, 1000.f, "%.2f");
                    ImGui::DragFloat("Grid Fade Distance##grid_dist", &render_engine->overlay->grid.fade_far, 0.01f, 0.f, 1000.f, "%.2f");
                    ImGui::DragFloat("min dist line width##min_dist_line_width", &render_engine->overlay->grid.max_dist_line_width, 0.01f, 0.f, 1000.f, "%.2f");
                    ImGui::DragFloat("max dist_line width##max_dist_line_width", &render_engine->overlay->grid.min_dist_line_width, 0.01f, 0.f, 1000.f, "%.2f");

                    ImGui::ColorEdit4("Color#color", (float*)&render_engine->overlay->grid.color);

                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Wireframe"))
                {
                    ImGui::Checkbox("Enable", &render_engine->overlay->enable_wireframe);
                    ImGui::ColorEdit4("Color#color", (float*)&render_engine->overlay->wireframe.color);
                    ImGui::DragFloat("Thickness##thickness", &render_engine->overlay->wireframe.thickness, 0.01f, 0.f, 1000.f, "%.2f");

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

            //if ((int)render_mode != render_mode_before)
                //buffer_reset = true;

            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }
}
/*
void RenderWindow::raster_props()
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

void RenderWindow::pathtracing_props()
{
    /*
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
    }    ImGui::Text("Path Tracing");

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
    */
//}

std::shared_ptr<Scene> RenderWindow::get_scene()
{
	return scene;
}

std::shared_ptr<Camera> RenderWindow::get_camera()
{
    return camera;
}

std::shared_ptr<Film> RenderWindow::get_film()
{
    return film;
}
