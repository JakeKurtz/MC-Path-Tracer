#include "Window.h"

void Window::frame_time()
{
    current_frame_time = glfwGetTime();
    delta_time = current_frame_time - last_frame_time;
    last_frame_time = current_frame_time;
}

void Window::update_window_properties()
{
    window_size = ImGui::GetWindowSize();

    window_min_pos = ImGui::GetWindowContentRegionMin();
    window_max_pos = ImGui::GetWindowContentRegionMax();

    window_pos = ImGui::GetWindowPos();

    window_center.x = (window_size.x / 2.f) + ImGui::GetWindowPos().x;
    window_center.y = (window_size.y / 2.f) + ImGui::GetWindowPos().y;
}

void Window::update_context_interaction_state()
{
    bool title_bar_hovered = ImGui::IsItemHovered();
    currently_focused = (ImGui::IsWindowFocused() && !title_bar_hovered) ? true : false;
}
