import torch
import time
from imgui_bundle import imgui
import imageio
import os
from tqdm import tqdm
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget
from renderer.gaussian_renderer import render_simple
from scene.cameras import CustomCam
import numpy as np
from splatviz_utils.cam_utils import shift_camera_by_baseline

class PoseEstimationTrainingDataWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Pose Estimation Stereo Training Data")
        self.is_recording = False
        self.circle_size = 20  # Diameter of the circle
        self.square_size = self.circle_size / 2**0.5  # Side length of inscribed square
        self.border_size = self.circle_size + 4  # Size of the white border circle
        self.square_rounding = 3.0  # Rounding radius for the square corners
        self.start_time = None
        self.last_record_time = None
        self.recording_fps = 15  # recording at 15Hz by default
        self.record_interval = 1.0 / self.recording_fps
        self.max_frames = 60 * 60 * 10  # 10 minutes at max 60Hz
        self.recorded_poses = None  # Will store the recorded poses
        self.current_frame = 0
        # Default rendering settings
        self.resolution_x = 1920
        self.resolution_y = 1080
        self.fov_x = 66
        self.fov_y = 40
        self.baseline_mm = 63
        self.model_unit_mm = round(235 / 0.791, ndigits=3)
        # Progress tracking
        self.show_progress = False  # New flag to control progress bar visibility
        self.render_progress = 0.0
        self.render_status = ""

    def add_labeled_input_int(self, label, value, width):
        imgui.set_next_item_width(width)
        changed, new_value = imgui.input_text(
            "##" + label,
            str(value),
            flags=imgui.InputTextFlags_.chars_decimal
        )
        if changed:
            try:
                return True, int(new_value)
            except ValueError:
                return False, value
        return False, value

    def __call__(self, show=True):
        if show:
            if imgui.begin_table("pose_recording_layout", 2, imgui.TableFlags_.borders_inner_v):
                # First column: Recording controls
                imgui.table_next_column()
                
                # Calculate total content width for centering in the column
                text_size = imgui.calc_text_size("Pose Recording")
                column_width = imgui.get_column_width()
                
                # Center the group horizontally in the column
                imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + (column_width - text_size.x) * 0.5)
                
                # Begin a group for vertical layout
                imgui.begin_group()
                
                # Centered text
                imgui.text("Pose Recording")
                
                # Add spacing to lower the button
                imgui.dummy(imgui.ImVec2(0, 10))
                
                # Center the button below the text
                button_x = (text_size.x - self.border_size) * 0.5
                imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + button_x)
                
                cursor_pos = imgui.get_cursor_pos()
                
                imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.8, 0.2, 0.2, 1.0))  # Red
                imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.9, 0.3, 0.3, 1.0))
                imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.7, 0.1, 0.1, 1.0))
                
                # Create circle button when not recording, square button when recording
                if self.is_recording:
                    # Record camera parameters at specified fps
                    current_time = time.time()
                    if self.last_record_time is None or (current_time - self.last_record_time) >= self.record_interval:
                        if self.current_frame < self.max_frames:
                            # Copy current pose to pre-allocated tensor
                            self.recorded_poses[self.current_frame].copy_(self.viz.args.cam_params)
                            self.current_frame += 1
                            self.last_record_time = current_time

                    imgui.push_style_var(imgui.StyleVar_.frame_rounding, self.square_rounding)  # Slightly rounded square when recording
                    padding = (self.circle_size - self.square_size) / 2
                    imgui.set_cursor_pos(imgui.ImVec2(cursor_pos.x + padding + 2, cursor_pos.y + padding + 2))
                    if imgui.button("##", imgui.ImVec2(self.square_size, self.square_size)):
                        self.is_recording = False
                        self.start_time = None
                        self.last_record_time = None
                        # Get only the recorded frames
                        self.recorded_poses = self.recorded_poses[:self.current_frame]
                        print(f"Recording finished. Recorded {self.current_frame} frames at {self.recording_fps}Hz")
                else:
                    imgui.push_style_var(imgui.StyleVar_.frame_rounding, self.circle_size/2)  # Circle when not recording
                    imgui.set_cursor_pos(imgui.ImVec2(cursor_pos.x + 2, cursor_pos.y + 2))
                    if imgui.button("##", imgui.ImVec2(self.circle_size, self.circle_size)):
                        self.is_recording = True
                        self.start_time = time.time()
                        self.last_record_time = None
                        self.current_frame = 0
                        # Pre-allocate tensor for recording
                        self.recorded_poses = torch.empty(
                            (self.max_frames, 4, 4), 
                            dtype=self.viz.args.cam_params.dtype,
                            device=self.viz.args.cam_params.device
                        )
                
                imgui.pop_style_var(1)  # Pop the rounding style
                imgui.pop_style_color(3)  # Pop the red button colors

                draw_list = imgui.get_window_draw_list()
                window_pos = imgui.get_window_pos()
                center = imgui.ImVec2(
                    window_pos.x + cursor_pos.x + self.border_size/2,
                    window_pos.y + cursor_pos.y + self.border_size/2
                )
                # White color in RGBA format (0xFFFFFFFF)
                draw_list.add_circle(
                    center=center,
                    radius=self.border_size/2,
                    col=0xFFFFFFFF,
                    thickness=1.0
                )
                
                # Create a fixed height area for the timer
                timer_area_height = 30  # Fixed height for timer area
                imgui.dummy(imgui.ImVec2(0, 5))  # Small spacing after button
                
                # Show timer when recording
                if self.is_recording and self.start_time is not None:
                    elapsed = time.time() - self.start_time
                    minutes = int(elapsed // 60)
                    seconds = int(elapsed % 60)
                    timer_text = f"{minutes:02d}:{seconds:02d}"
                    
                    # Calculate timer text position
                    timer_size = imgui.calc_text_size(timer_text)
                    timer_x = cursor_pos.x + (self.border_size - timer_size.x) * 0.5
                    timer_y = cursor_pos.y + self.border_size + 5  # 5 pixels below the button
                    
                    # Draw white timer text
                    draw_list.add_text(
                        imgui.ImVec2(window_pos.x + timer_x, window_pos.y + timer_y),
                        0xFFFFFFFF,
                        timer_text
                    )
                
                # Add remaining spacing to maintain fixed height
                current_y = imgui.get_cursor_pos().y
                target_y = cursor_pos.y + self.border_size + timer_area_height
                remaining_space = target_y - current_y
                if remaining_space > 0:
                    imgui.dummy(imgui.ImVec2(0, remaining_space))
                
                # Center the frame rate controls in a new group
                imgui.begin_group()
                
                # Center "Frame Rate" label relative to "Pose Recording" width
                fps_text = "Frame Rate"
                fps_text_size = imgui.calc_text_size(fps_text)
                label_x = (text_size.x - fps_text_size.x) * 0.5
                imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + label_x)
                imgui.text(fps_text)
                imgui.spacing()
                
                # Center the slider relative to "Pose Recording" width, but make it twice as wide
                slider_width = text_size.x * 1.6  # Make slider 160% of the title text width
                slider_x = (text_size.x - slider_width) * 0.5
                imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + slider_x)
                imgui.set_next_item_width(slider_width)
                
                # Only allow changing frame rate when not recording
                if not self.is_recording:
                    changed, new_fps = imgui.slider_int(
                        "##fps", 
                        self.recording_fps,
                        1,  # v_min
                        60,  # v_max
                        "%d Hz"  # format
                    )
                    if changed:
                        self.recording_fps = new_fps
                        self.record_interval = 1.0 / self.recording_fps
                else:
                    # Show disabled slider during recording
                    imgui.begin_disabled()
                    imgui.slider_int(
                        "##fps", 
                        self.recording_fps,
                        1,  # v_min
                        60,  # v_max
                        "%d Hz"  # format
                    )
                    imgui.end_disabled()
                
                imgui.end_group()
                
                imgui.end_group()
                
                # Second column: Camera Parameters Matrix
                imgui.table_next_column()
                
                # Center the title
                title_text = "Extrinsic Matrix / Pose"
                title_size = imgui.calc_text_size(title_text)
                column_width = imgui.get_column_width()
                imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + (column_width - title_size.x) * 0.5)
                imgui.text(title_text)
                imgui.spacing()
                
                # Calculate total width of one matrix row (4 numbers with spacing)
                number_width = imgui.calc_text_size("  0.000").x  # Width of one number with padding
                spacing_width = imgui.calc_text_size("  ").x  # Width of spacing between numbers
                matrix_width = number_width * 4 + spacing_width * 3  # Total width of one row
                
                # Display 4x4 matrix with formatting
                cam_params = self.viz.args.cam_params.cpu().numpy()
                for i in range(4):
                    # Center each row
                    imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + (column_width - matrix_width) * 0.5)
                    for j in range(4):
                        if j > 0:
                            imgui.same_line()
                        imgui.text(f"{cam_params[i,j]:7.3f}")
                
                imgui.end_table()
                
                # Add horizontal separator after the table with more prominence
                imgui.spacing()
                imgui.spacing()
                imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(0, 0))
                imgui.push_style_color(imgui.Col_.separator, imgui.ImVec4(0.4, 0.4, 0.4, 1.0))  # Slightly lighter than default
                imgui.separator()
                imgui.separator()  # Double separator for thickness
                imgui.pop_style_color(1)
                imgui.pop_style_var(1)
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()
                
                # Add label showing number of recorded poses
                num_poses_text = f"Number of recorded poses: {self.current_frame}"
                text_size = imgui.calc_text_size(num_poses_text)
                window_width = imgui.get_window_size().x
                imgui.set_cursor_pos_x((window_width - text_size.x) * 0.5)
                imgui.text(num_poses_text)
                
                imgui.spacing()
                
                # Add centered button
                button_text = "Render and Save Dataset"
                button_width = imgui.calc_text_size(button_text).x + 40  # Add some padding
                imgui.set_cursor_pos_x((window_width - button_width) * 0.5)
                
                if imgui.button(button_text, imgui.ImVec2(button_width, 0)):
                    save_dir = "/home/chris/Dev/CAREER/pose_estimation_dataset"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Get the correct dataset index by checking existing files
                    existing_indices = []
                    for filename in os.listdir(save_dir):
                        try:
                            idx = int(filename.split("_")[2])
                            existing_indices.append(idx)
                        except (IndexError, ValueError):
                            continue
                    
                    # Get next available index
                    next_idx = 0 if not existing_indices else max(existing_indices) + 1
                    
                    # Create directories for left and right images
                    left_img_dir = os.path.join(save_dir, f"left_images_{next_idx}")
                    right_img_dir = os.path.join(save_dir, f"right_images_{next_idx}")
                    os.makedirs(left_img_dir, exist_ok=True)
                    os.makedirs(right_img_dir, exist_ok=True)
                    
                    # Get models and ensure they're on GPU
                    left_model = self.viz.renderer.renderer.gaussian_models[0]
                    right_model = self.viz.renderer.renderer.gaussian_models[1]
                    bg_color = self.viz.args.background_color.to("cuda")
                    
                    total_frames = self.recorded_poses.shape[0]
                    self.show_progress = True  # Show progress bar
                    self.render_progress = 0.0
                    self.render_status = "Starting render..."
                    
                    # Start rendering in next frame
                    self.render_state = {
                        "left_dir": left_img_dir,
                        "right_dir": right_img_dir,
                        "left_model": left_model,
                        "right_model": right_model,
                        "bg_color": bg_color,
                        "total_frames": total_frames,
                        "current_frame": 0,
                        "save_dir": save_dir,
                        "dataset_idx": next_idx
                    }
                    
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()
                
                # Create four columns for settings
                if imgui.begin_table("settings", 4, imgui.TableFlags_.none):
                    # Headers with bigger text
                    imgui.table_next_column()
                    column_width = imgui.get_column_width()
                    text_size = imgui.calc_text_size("Resolution")
                    imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + (column_width - text_size.x) * 0.5)
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))  # White text
                    imgui.text("Resolution")
                    imgui.pop_style_color()
                    
                    imgui.table_next_column()
                    column_width = imgui.get_column_width()
                    text_size = imgui.calc_text_size("FOV")
                    imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + (column_width - text_size.x) * 0.5)
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))  # White text
                    imgui.text("FOV")
                    imgui.pop_style_color()
                    
                    imgui.table_next_column()
                    column_width = imgui.get_column_width()
                    text_size = imgui.calc_text_size("Baseline (mm)")
                    imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + (column_width - text_size.x) * 0.5)
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))  # White text
                    imgui.text("Baseline (mm)")
                    imgui.pop_style_color()
                    
                    imgui.table_next_column()
                    column_width = imgui.get_column_width()
                    text_size = imgui.calc_text_size("Model unit (mm)")
                    imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + (column_width - text_size.x) * 0.5)
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))  # White text
                    imgui.text("Model unit (mm)")
                    imgui.pop_style_color()
                    
                    # Add extra vertical spacing after titles
                    imgui.table_next_row()
                    imgui.dummy(imgui.ImVec2(0, 12))  # 12 pixels of vertical spacing
                    
                    # Input fields row
                    imgui.table_next_row()
                    
                    # Resolution column
                    imgui.table_next_column()
                    input_width = 70
                    # Center the X/Y fields in the column
                    total_width = input_width * 2 + imgui.calc_text_size("X: Y:").x + 15  # 15 for spacing
                    imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + (column_width - total_width) * 0.5)
                    imgui.align_text_to_frame_padding()  # Align text with input field
                    imgui.text("X:")
                    imgui.same_line(spacing=5)
                    changed_x, new_res_x = self.add_labeled_input_int("resx", self.resolution_x, input_width)
                    if changed_x and 480 <= new_res_x <= 3840:
                        self.resolution_x = new_res_x
                    self.viz.args.resolution_x = self.resolution_x
                    imgui.same_line(spacing=5)
                    imgui.align_text_to_frame_padding()  # Align text with input field
                    imgui.text("Y:")
                    imgui.same_line(spacing=5)
                    changed_y, new_res_y = self.add_labeled_input_int("resy", self.resolution_y, input_width)
                    if changed_y and 480 <= new_res_y <= 2160:
                        self.resolution_y = new_res_y
                    self.viz.args.resolution_y = self.resolution_y
                    
                    # FOV column
                    imgui.table_next_column()
                    # Center the X/Y fields in the column
                    imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + (column_width - total_width) * 0.5)
                    imgui.align_text_to_frame_padding()  # Align text with input field
                    imgui.text("X:")
                    imgui.same_line(spacing=5)
                    changed_fov_x, new_fov_x = self.add_labeled_input_int("fovx", self.fov_x, input_width)
                    if changed_fov_x and 10 <= new_fov_x <= 120:
                        self.fov_x = new_fov_x
                    self.viz.args.fov_x = self.fov_x
                    imgui.same_line(spacing=5)
                    imgui.align_text_to_frame_padding()  # Align text with input field
                    imgui.text("Y:")
                    imgui.same_line(spacing=5)
                    changed_fov_y, new_fov_y = self.add_labeled_input_int("fovy", self.fov_y, input_width)
                    if changed_fov_y and 10 <= new_fov_y <= 120:
                        self.fov_y = new_fov_y
                    self.viz.args.fov_y = self.fov_y
                    
                    # Baseline column - single centered input
                    imgui.table_next_column()
                    column_width = imgui.get_column_width()
                    imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + (column_width - input_width) * 0.5)
                    changed_dist, new_baseline_mm = self.add_labeled_input_int("baseline_dist", self.baseline_mm, input_width)
                    if changed_dist and 0 <= new_baseline_mm <= 1000:
                        self.baseline_mm = new_baseline_mm
                    self.viz.args.baseline_mm = self.baseline_mm
                    
                    # Model unit column - single centered input
                    imgui.table_next_column()
                    column_width = imgui.get_column_width()
                    imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + (column_width - input_width) * 0.5)
                    changed_unit, new_model_unit_mm = self.add_labeled_input_int("baseline_unit", self.model_unit_mm, input_width)
                    if changed_unit and 0 <= new_model_unit_mm <= 1000:
                        self.model_unit_mm = new_model_unit_mm
                    self.viz.args.model_unit_mm = self.model_unit_mm
                    
                    imgui.end_table()
                
                # Add progress bar if it should be shown
                if self.show_progress:
                    imgui.spacing()
                    imgui.spacing()
                    
                    # Center the progress text above the bar
                    text_size = imgui.calc_text_size(self.render_status)
                    window_width = imgui.get_window_size().x
                    imgui.set_cursor_pos_x((window_width - text_size.x) * 0.5)
                    imgui.text(self.render_status)
                    
                    imgui.spacing()
                    
                    # Center and size the progress bar
                    progress_width = window_width * 0.8  # 80% of window width
                    imgui.set_cursor_pos_x((window_width - progress_width) * 0.5)
                    imgui.push_style_color(imgui.Col_.plot_histogram, imgui.ImVec4(0.2, 0.8, 0.2, 1.0))  # Green progress
                    imgui.progress_bar(
                        self.render_progress,
                        imgui.ImVec2(progress_width, 0),
                        ""  # Empty string since we're showing text above
                    )
                    imgui.pop_style_color(1)
                    
                    # Process next frame if rendering is in progress
                    if hasattr(self, "render_state"):
                        if self.render_state["current_frame"] < self.render_state["total_frames"]:
                            i = self.render_state["current_frame"]
                            self.render_progress = i / self.render_state["total_frames"]
                            self.render_status = f"Rendering frame {i+1}/{self.render_state['total_frames']}"
                            
                            # Ensure pose is on GPU
                            pose = self.recorded_poses[i].to("cuda")
                            
                            # Save pose in the left images directory with pose_ prefix
                            np.save(os.path.join(self.render_state["left_dir"], f"pose_{i:06d}.npy"), pose.cpu().numpy())
                            
                            # Render left view
                            left_render_cam = CustomCam(
                                width=1920,
                                height=1080,
                                fovy=self.fov_y / 360 * 2 * np.pi,
                                fovx=self.fov_x / 360 * 2 * np.pi,
                                znear=0.01,
                                zfar=100,
                                extr=pose,
                            )
                            with torch.cuda.amp.autocast():
                                render = render_simple(viewpoint_camera=left_render_cam, pc=self.render_state["left_model"], bg_color=self.render_state["bg_color"])
                                img = render["render"]
                                depth = render["depth"] / render["depth"].max()
                                img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                                depth = (depth * 255).clamp(0, 255).squeeze().to(torch.uint8).cpu().numpy()
                                imageio.imwrite(os.path.join(self.render_state["left_dir"], f"{i:06d}.jpeg"), img)
                                imageio.imwrite(os.path.join(self.render_state["left_dir"], f"{i:06d}_depth.jpeg"), depth)

                            # Render right view
                            right_pose = shift_camera_by_baseline(pose, self.baseline_mm, self.model_unit_mm)
                            np.save(os.path.join(self.render_state["right_dir"], f"pose_{i:06d}.npy"), right_pose.cpu().numpy())
                            right_render_cam = CustomCam(
                                width=1920,  # Fixed width
                                height=1080,  # Fixed height - no padding needed for PNG
                                fovy=self.fov_y / 360 * 2 * np.pi,
                                fovx=self.fov_x / 360 * 2 * np.pi,
                                znear=0.01,
                                zfar=100,
                                extr=right_pose,
                            )
                            with torch.cuda.amp.autocast():
                                render = render_simple(viewpoint_camera=right_render_cam, pc=self.render_state["right_model"], bg_color=self.render_state["bg_color"])
                                img = render["render"]
                                depth = render["depth"] / render["depth"].max()
                                img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                                depth = (depth * 255).clamp(0, 255).squeeze().to(torch.uint8).cpu().numpy()
                                imageio.imwrite(os.path.join(self.render_state["right_dir"], f"{i:06d}.jpeg"), img)
                                imageio.imwrite(os.path.join(self.render_state["right_dir"], f"{i:06d}_depth.jpeg"), depth)
                            
                            self.render_state["current_frame"] += 1
                        else:
                            self.render_progress = 1.0
                            self.render_status = "Rendering complete!"
                            del self.render_state  # Cleanup but keep progress bar visible
