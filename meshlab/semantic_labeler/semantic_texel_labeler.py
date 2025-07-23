#!/usr/bin/env python3
"""
Semantic Texel Labeler for Minecraft Character Skins
A GUI tool for creating semantic labels on 64x64 minecraft character textures
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import json
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import os

class SemanticTexelLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Semantic Texel Labeler - Minecraft Character Skin")
        self.root.geometry("1200x800")

        # Data structures
        self.texture_size = 64  # 64x64 minecraft skin
        self.semantic_map = np.zeros((self.texture_size, self.texture_size), dtype=int)
        self.labels = {
            0: {"name": "None", "color": "#FFFFFF", "description": "No label"},
            1: {"name": "Skin", "color": "#FFDBAC", "description": "Character skin texture"},
            2: {"name": "Hair", "color": "#8B4513", "description": "Hair/fur texture"},
            3: {"name": "Clothing", "color": "#4169E1", "description": "Clothing fabric"},
            4: {"name": "Metal", "color": "#C0C0C0", "description": "Metal materials"},
            5: {"name": "Leather", "color": "#8B4513", "description": "Leather materials"}
        }
        self.current_label = 1
        self.painting = False
        self.zoom_factor = 8  # Display zoom

        # Drawing modes and state
        self.draw_mode = "freehand"  # or "rectangle"
        self.rect_start = None  # Starting point for rectangle
        self.rect_preview = None  # Canvas item for rectangle preview

        # Visualization modes
        self.show_only_current_label = False

        # Background texture
        self.background_texture = None
        self.background_image = None

        self.setup_ui()
        self.update_canvas()

    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Tools and labels
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Right panel - Canvas
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)

    def setup_left_panel(self, parent):
        """Setup the left control panel"""
        # File operations
        file_frame = ttk.LabelFrame(parent, text="File Operations")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="Load Background Texture",
                  command=self.load_background_texture).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Import Semantic Map",
                  command=self.import_semantic_map).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Export Semantic Map",
                  command=self.export_semantic_map).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Clear All Labels",
                  command=self.clear_all_labels).pack(fill=tk.X, pady=2)

        # Label management
        label_frame = ttk.LabelFrame(parent, text="Label Management")
        label_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Add/Remove buttons
        button_frame = ttk.Frame(label_frame)
        button_frame.pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Add Label",
                  command=self.add_label).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Remove Label",
                  command=self.remove_label).pack(side=tk.LEFT)

        # Label list
        self.label_listbox = tk.Listbox(label_frame, height=10)
        self.label_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.label_listbox.bind('<<ListboxSelect>>', self.on_label_select)

        # Edit label frame
        edit_frame = ttk.LabelFrame(label_frame, text="Edit Label")
        edit_frame.pack(fill=tk.X, pady=5)

        ttk.Label(edit_frame, text="Name:").pack(anchor=tk.W)
        self.label_name_var = tk.StringVar()
        self.label_name_entry = ttk.Entry(edit_frame, textvariable=self.label_name_var)
        self.label_name_entry.pack(fill=tk.X, pady=(0, 5))
        self.label_name_entry.bind('<Return>', self.update_label_name)

        ttk.Label(edit_frame, text="Description:").pack(anchor=tk.W)
        self.label_desc_var = tk.StringVar()
        self.label_desc_entry = ttk.Entry(edit_frame, textvariable=self.label_desc_var)
        self.label_desc_entry.pack(fill=tk.X, pady=(0, 5))
        self.label_desc_entry.bind('<Return>', self.update_label_description)

        color_frame = ttk.Frame(edit_frame)
        color_frame.pack(fill=tk.X)
        ttk.Label(color_frame, text="Color:").pack(side=tk.LEFT)
        self.color_button = tk.Button(color_frame, text="   ", width=3,
                                    command=self.change_label_color)
        self.color_button.pack(side=tk.LEFT, padx=(5, 0))

        # Drawing mode controls
        mode_frame = ttk.LabelFrame(parent, text="Drawing Mode")
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        self.mode_var = tk.StringVar(value="freehand")
        ttk.Radiobutton(mode_frame, text="Freehand", value="freehand",
                       variable=self.mode_var, command=self.change_draw_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Rectangle", value="rectangle",
                       variable=self.mode_var, command=self.change_draw_mode).pack(anchor=tk.W)

        # Zoom controls
        zoom_frame = ttk.LabelFrame(parent, text="View Controls")
        zoom_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(zoom_frame, text="Zoom:").pack(anchor=tk.W)
        zoom_control = ttk.Frame(zoom_frame)
        zoom_control.pack(fill=tk.X)
        ttk.Button(zoom_control, text="-", width=3,
                  command=self.zoom_out).pack(side=tk.LEFT)
        self.zoom_label = ttk.Label(zoom_control, text=f"{self.zoom_factor}x")
        self.zoom_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(zoom_control, text="+", width=3,
                  command=self.zoom_in).pack(side=tk.LEFT)

        # Visualization controls
        viz_frame = ttk.LabelFrame(parent, text="Visualization")
        viz_frame.pack(fill=tk.X, pady=(0, 10))

        self.show_current_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(viz_frame, text="Show only current label",
                       variable=self.show_current_var,
                       command=self.toggle_label_visibility).pack(anchor=tk.W)

        ttk.Button(viz_frame, text="Validate Unlabeled Texels",
                  command=self.validate_unlabeled_texels).pack(fill=tk.X, pady=2)

        # Info frame (moved after visualization controls)
        info_frame = ttk.LabelFrame(parent, text="Info")
        info_frame.pack(fill=tk.X)
        self.info_label = ttk.Label(info_frame, text="Click and drag to paint labels")
        self.info_label.pack(anchor=tk.W)

        self.update_label_list()

    def setup_right_panel(self, parent):
        """Setup the right canvas panel"""
        # Canvas frame with scrollbars
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        # Grid layout
        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")

        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_painting)
        self.canvas.bind("<B1-Motion>", self.paint_pixel)
        self.canvas.bind("<ButtonRelease-1>", self.stop_painting)
        self.canvas.bind("<Motion>", self.on_mouse_move)

    def update_label_list(self):
        """Update the label listbox"""
        self.label_listbox.delete(0, tk.END)
        for label_id, label_info in self.labels.items():
            self.label_listbox.insert(tk.END, f"{label_id}: {label_info['name']}")
            # Set background color for the last inserted item
            last_index = self.label_listbox.size() - 1
            self.label_listbox.itemconfig(last_index, background=label_info['color'])

    def on_label_select(self, event):
        """Handle label selection"""
        selection = self.label_listbox.curselection()
        if selection:
            index = selection[0]
            label_ids = list(self.labels.keys())
            if index < len(label_ids):
                self.current_label = label_ids[index]
                label_info = self.labels[self.current_label]
                self.label_name_var.set(label_info['name'])
                self.label_desc_var.set(label_info.get('description', ''))
                self.color_button.config(bg=label_info['color'])

    def add_label(self):
        """Add a new label"""
        # Find next available ID
        new_id = max(self.labels.keys()) + 1
        name = f"Label {new_id}"
        color = "#FF0000"  # Default red

        self.labels[new_id] = {
            "name": name,
            "color": color,
            "description": "New label description"
        }
        self.update_label_list()

        # Select the new label
        self.label_listbox.selection_set(len(self.labels) - 1)
        self.on_label_select(None)

    def remove_label(self):
        """Remove the selected label"""
        if self.current_label == 0:
            messagebox.showwarning("Warning", "Cannot remove the 'None' label")
            return

        if self.current_label in self.labels:
            # Replace all instances of this label with 'None' (0)
            self.semantic_map[self.semantic_map == self.current_label] = 0
            del self.labels[self.current_label]
            self.current_label = 0
            self.update_label_list()
            self.update_canvas()

    def update_label_name(self, event=None):
        """Update the current label's name"""
        if self.current_label in self.labels:
            self.labels[self.current_label]['name'] = self.label_name_var.get()
            self.update_label_list()

    def update_label_description(self, event=None):
        """Update the current label's description"""
        if self.current_label in self.labels:
            self.labels[self.current_label]['description'] = self.label_desc_var.get()

    def change_label_color(self):
        """Change the current label's color"""
        if self.current_label in self.labels:
            color = colorchooser.askcolor(self.labels[self.current_label]['color'])
            if color[1]:  # If a color was selected
                self.labels[self.current_label]['color'] = color[1]
                self.color_button.config(bg=color[1])
                self.update_label_list()
                self.update_canvas()

    def change_draw_mode(self):
        """Change the drawing mode between freehand and rectangle"""
        self.draw_mode = self.mode_var.get()
        if self.rect_preview:
            self.canvas.delete(self.rect_preview)
            self.rect_preview = None
        self.rect_start = None

    def zoom_in(self):
        """Increase zoom factor"""
        if self.zoom_factor < 16:
            self.zoom_factor += 1
            self.zoom_label.config(text=f"{self.zoom_factor}x")
            self.update_canvas()

    def zoom_out(self):
        """Decrease zoom factor"""
        if self.zoom_factor > 1:
            self.zoom_factor -= 1
            self.zoom_label.config(text=f"{self.zoom_factor}x")
            self.update_canvas()

    def toggle_label_visibility(self):
        """Toggle visibility of non-current labels"""
        self.show_only_current_label = self.show_current_var.get()
        self.update_canvas()

    def validate_unlabeled_texels(self):
        """Check for unlabeled non-transparent texels"""
        if not self.background_image:
            messagebox.showwarning("Warning", "Please load a background texture first")
            return

        # Get alpha channel
        alpha = np.array(self.background_image.getchannel('A'))

        # Find non-transparent texels (alpha > 0) that aren't labeled
        unlabeled_mask = (alpha > 0) & (self.semantic_map == 0)
        unlabeled_count = np.sum(unlabeled_mask)

        if unlabeled_count == 0:
            messagebox.showinfo("Validation", "All non-transparent texels are labeled!")
            return

        # Create validation label if it doesn't exist
        validation_id = None
        for label_id, info in self.labels.items():
            if info['name'] == "validation-check":
                validation_id = label_id
                break

        if validation_id is None:
            validation_id = max(self.labels.keys()) + 1
            self.labels[validation_id] = {
                "name": "validation-check",
                "color": "#FF00FF"  # Magenta for visibility
            }
            self.update_label_list()

        # Mark unlabeled texels
        self.semantic_map[unlabeled_mask] = validation_id
        self.update_canvas()

        # Report to user
        messagebox.showwarning(
            "Validation Results",
            f"Found {unlabeled_count} unlabeled non-transparent texels.\n"
            "These have been marked with the 'validation-check' label (magenta).\n"
            "Please label these areas appropriately."
        )

    def update_canvas(self):
        """Update the canvas display"""
        self.canvas.delete("all")

        # Create image
        display_size = self.texture_size * self.zoom_factor

        # Create base image (background texture or white)
        if self.background_image:
            base_img = self.background_image.resize((display_size, display_size), Image.NEAREST)
        else:
            base_img = Image.new('RGB', (display_size, display_size), 'white')

        # Create overlay for semantic labels
        overlay = Image.new('RGBA', (display_size, display_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw semantic labels
        for y in range(self.texture_size):
            for x in range(self.texture_size):
                label_id = self.semantic_map[y, x]
                if label_id != 0:  # Skip 'None' labels
                    # Skip if showing only current label and this isn't it
                    if self.show_only_current_label and label_id != self.current_label:
                        continue

                    color = self.labels[label_id]['color']
                    # Convert hex to RGB and add alpha
                    rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

                    # Make validation-check more visible
                    if self.labels[label_id]['name'] == "validation-check":
                        rgba = rgb + (192,)  # More opaque for validation
                    else:
                        rgba = rgb + (128,)  # Semi-transparent for normal labels

                    # Draw rectangle for this pixel
                    x1 = x * self.zoom_factor
                    y1 = y * self.zoom_factor
                    x2 = x1 + self.zoom_factor
                    y2 = y1 + self.zoom_factor
                    draw.rectangle([x1, y1, x2-1, y2-1], fill=rgba)

        # Combine base and overlay
        if self.background_image:
            base_img = Image.alpha_composite(base_img.convert('RGBA'), overlay)
        else:
            base_img = overlay

        # Draw grid
        draw = ImageDraw.Draw(base_img)
        grid_color = (128, 128, 128, 64) if self.background_image else (200, 200, 200, 255)
        for i in range(0, display_size + 1, self.zoom_factor):
            draw.line([(i, 0), (i, display_size)], fill=grid_color, width=1)
            draw.line([(0, i), (display_size, i)], fill=grid_color, width=1)

        # Convert to PhotoImage and display
        self.canvas_image = ImageTk.PhotoImage(base_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)

        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def get_pixel_coords(self, event):
        """Convert canvas coordinates to pixel coordinates"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        pixel_x = int(canvas_x // self.zoom_factor)
        pixel_y = int(canvas_y // self.zoom_factor)
        return pixel_x, pixel_y

    def start_painting(self, event):
        """Start painting mode"""
        pixel_x, pixel_y = self.get_pixel_coords(event)
        if not (0 <= pixel_x < self.texture_size and 0 <= pixel_y < self.texture_size):
            return

        if self.draw_mode == "freehand":
            self.painting = True
            self.paint_pixel(event)
        else:  # rectangle mode
            self.rect_start = (pixel_x, pixel_y)

    def paint_pixel(self, event):
        """Paint a pixel with the current label"""
        if self.draw_mode == "freehand":
            if not self.painting:
                return

            pixel_x, pixel_y = self.get_pixel_coords(event)
            if 0 <= pixel_x < self.texture_size and 0 <= pixel_y < self.texture_size:
                self.semantic_map[pixel_y, pixel_x] = self.current_label
                self.update_canvas()
        else:  # rectangle mode
            if not self.rect_start:
                return

            # Get current mouse position
            pixel_x, pixel_y = self.get_pixel_coords(event)
            if 0 <= pixel_x < self.texture_size and 0 <= pixel_y < self.texture_size:
                # Delete previous preview
                if self.rect_preview:
                    self.canvas.delete(self.rect_preview)

                # Draw new preview rectangle
                x1 = self.rect_start[0] * self.zoom_factor
                y1 = self.rect_start[1] * self.zoom_factor
                x2 = (pixel_x + 1) * self.zoom_factor
                y2 = (pixel_y + 1) * self.zoom_factor

                self.rect_preview = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline=self.labels[self.current_label]['color'],
                    width=2
                )

    def stop_painting(self, event):
        """Stop painting mode"""
        if self.draw_mode == "freehand":
            self.painting = False
        else:  # rectangle mode
            if self.rect_start:
                end_x, end_y = self.get_pixel_coords(event)
                if 0 <= end_x < self.texture_size and 0 <= end_y < self.texture_size:
                    # Get rectangle bounds
                    start_x, start_y = self.rect_start
                    min_x = min(start_x, end_x)
                    max_x = max(start_x, end_x)
                    min_y = min(start_y, end_y)
                    max_y = max(start_y, end_y)

                    # Fill the rectangle
                    self.semantic_map[min_y:max_y+1, min_x:max_x+1] = self.current_label

                    # Clean up
                    if self.rect_preview:
                        self.canvas.delete(self.rect_preview)
                    self.rect_preview = None
                    self.rect_start = None
                    self.update_canvas()

    def on_mouse_move(self, event):
        """Handle mouse movement for info display"""
        pixel_x, pixel_y = self.get_pixel_coords(event)
        if 0 <= pixel_x < self.texture_size and 0 <= pixel_y < self.texture_size:
            label_id = self.semantic_map[pixel_y, pixel_x]
            label_info = self.labels.get(label_id, {})
            label_name = label_info.get('name', 'Unknown')
            label_desc = label_info.get('description', '')
            self.info_label.config(text=f"Pixel ({pixel_x}, {pixel_y}): {label_name} - {label_desc}")
        else:
            self.info_label.config(text="Click and drag to paint labels")

    def load_background_texture(self):
        """Load a background texture image"""
        filename = filedialog.askopenfilename(
            title="Select Background Texture",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if filename:
            try:
                img = Image.open(filename)
                # Resize to 64x64 if needed
                if img.size != (64, 64):
                    img = img.resize((64, 64), Image.NEAREST)
                    messagebox.showinfo("Info", f"Image resized from {Image.open(filename).size} to 64x64")
                self.background_image = img.convert('RGBA')
                self.update_canvas()
                messagebox.showinfo("Success", "Background texture loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def clear_all_labels(self):
        """Clear all labels from the semantic map"""
        if messagebox.askyesno("Confirm", "Clear all labels? This cannot be undone."):
            self.semantic_map.fill(0)
            self.update_canvas()

    def export_semantic_map(self):
        """Export the semantic map to JSON"""
        filename = filedialog.asksaveasfilename(
            title="Export Semantic Map",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            try:
                data = {
                    "metadata": {
                        "texture_size": self.texture_size,
                        "version": "1.0",
                        "description": "Minecraft character skin semantic map"
                    },
                    "labels": self.labels,
                    "semantic_map": self.semantic_map.tolist()
                }
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                messagebox.showinfo("Success", f"Semantic map exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")

    def import_semantic_map(self):
        """Import a semantic map from JSON"""
        filename = filedialog.askopenfilename(
            title="Import Semantic Map",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)

                # Validate data structure
                if 'labels' not in data or 'semantic_map' not in data:
                    raise ValueError("Invalid semantic map file format")

                # Convert label IDs to integers and ensure description field exists
                self.labels = {}
                for k, v in data['labels'].items():
                    label_id = int(k)
                    self.labels[label_id] = {
                        "name": v["name"],
                        "color": v["color"],
                        "description": v.get("description", "No description")
                    }

                self.semantic_map = np.array(data['semantic_map'], dtype=int)

                if self.semantic_map.shape != (self.texture_size, self.texture_size):
                    raise ValueError(f"Invalid semantic map size: {self.semantic_map.shape}")

                self.update_label_list()
                self.update_canvas()
                messagebox.showinfo("Success", f"Semantic map imported from {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to import: {str(e)}")

def main():
    root = tk.Tk()
    app = SemanticTexelLabeler(root)
    root.mainloop()

if __name__ == "__main__":
    main()