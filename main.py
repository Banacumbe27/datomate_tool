import os
import queue
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class YoloAnnotatorApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Datomate YOLO Annotator")
        self.geometry("1400x900")
        self.minsize(960, 640)

        self._initialize_state()
        self._configure_layout()
        self._create_main_frames()
        self._build_left_panel()
        self._build_right_panel()
        self.protocol("WM_DELETE_WINDOW", self._on_app_close)
        self.background_result_after_id = self.after(50, self._poll_background_results)
        self.bind_all("<Control-z>", self._on_undo_shortcut)
        self.bind_all("<Control-y>", self._on_redo_shortcut)
        self.bind_all("<Control-Z>", self._on_redo_shortcut)
        self.bind_all("<Control-c>", self._on_copy_shortcut)
        self.bind_all("<Control-v>", self._on_paste_shortcut)
        self.bind_all("<Control-d>", self._on_duplicate_shortcut)
        self.bind_all("<Escape>", self._on_escape_pressed)
        self.bind_all("<BackSpace>", self._on_delete_selected)
        self.bind_all("<Delete>", self._on_delete_selected)
        self.bind_all("<Control_L>", self._on_ctrl_press)
        self.bind_all("<Control_R>", self._on_ctrl_press)
        self.bind_all("<KeyRelease-Control_L>", self._on_ctrl_release)
        self.bind_all("<KeyRelease-Control_R>", self._on_ctrl_release)
        self.bind_all("<KeyPress-space>", self._on_spacebar_press)
        self.bind_all("<KeyRelease-space>", self._on_spacebar_release)
        self.bind_all("<Button-1>", self._on_global_left_click, add="+")

    def _initialize_state(self) -> None:
        self.dataset_path: Path | None = None
        self.dataset_images_root: Path | None = None
        self.dataset_labels_root: Path | None = None
        self.image_filenames: list[str] = []
        self.selected_image_filename: str | None = None
        self.default_class_id: int = 0
        self.current_annotations: list[dict[str, int | float]] = []
        self.image_entries: list[dict[str, object]] = []
        self.filtered_image_entries: list[dict[str, object]] = []
        self.image_path_lookup: dict[str, Path] = {}
        self.image_tree_item_ids: dict[str, str] = {}
        self.image_tree_selection_guard: bool = False
        self.background_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="datomate")
        self.background_result_queue: queue.Queue = queue.Queue()
        self.dataset_scan_token: int = 0
        self.image_list_refresh_token: int = 0
        self.image_load_token: int = 0
        self.image_list_refresh_after_id: str | None = None
        self.image_list_render_after_id: str | None = None
        self.pending_image_list_entries: list[dict[str, object]] = []
        self.pending_image_list_index: int = 0
        self.list_loading_after_id: str | None = None
        self.list_loading_message: str = ""
        self.list_loading_phase: int = 0
        self.list_loading_label: ctk.CTkLabel | None = None
        self.canvas_loading_after_id: str | None = None
        self.canvas_loading_message: str = ""
        self.canvas_loading_phase: int = 0
        self.canvas_loading_text_id: int | None = None
        self.annotation_colors: tuple[str, ...] = (
            "#22c55e",
            "#f97316",
            "#38bdf8",
            "#eab308",
            "#a855f7",
            "#ef4444",
        )
        self.canvas_image_ref: ImageTk.PhotoImage | None = None
        self.canvas_image_item_id: int | None = None
        self.canvas_image_render_bounds: dict[str, float] = {
            "left": 0.0,
            "top": 0.0,
            "right": 0.0,
            "bottom": 0.0,
        }
        self.current_image_path: Path | None = None
        self.current_label_path: Path | None = None
        self.current_pil_image: Image.Image | None = None
        self.current_preview_levels: list[tuple[int, Image.Image]] = []
        self.canvas_display_state: dict[str, float] = {
            "offset_x": 0.0,
            "offset_y": 0.0,
            "display_width": 1.0,
            "display_height": 1.0,
            "image_width": 1.0,
            "image_height": 1.0,
        }
        self.current_tool: str = "draw"
        self.is_drawing: bool = False
        self.is_dragging: bool = False
        self.is_panning: bool = False
        self.is_selecting_region: bool = False
        self.drag_data: dict[str, object] = self._create_empty_drag_data()
        self.pan_data: dict[str, float] = self._create_empty_pan_data()
        self.selection_rect_id: int | None = None
        self.selected_annotation_id: int | None = None
        self.selected_annotation_ids: set[int] = set()
        self.temp_rectangle_id: int | None = None
        self.annotation_canvas_items: dict[int, dict[str, int]] = {}
        self.canvas_item_to_annotation_id: dict[int, int] = {}
        self.selection_handle_items: dict[tuple[int, str], int] = {}
        self.handle_item_to_name: dict[int, tuple[int, str]] = {}
        self.resize_tolerance: int = 10
        self.minimum_box_size: int = 6
        self.handle_radius: int = 5
        self.annotation_id_counter: int = 0
        self.zoom_factor: float = 1.0
        self.min_zoom_factor: float = 1.0
        self.max_zoom_factor: float = 32.0
        self.pan_offset_x: float = 0.0
        self.pan_offset_y: float = 0.0
        self.pending_render_after_id: str | None = None
        self.pending_force_rebuild: bool = False
        self.pending_interactive_render: bool = False
        self.high_quality_render_after_id: str | None = None
        self.undo_stack: list[dict[str, object]] = []
        self.redo_stack: list[dict[str, object]] = []
        self.annotation_clipboard: list[dict[str, float | int]] = []
        self.context_menu_canvas_x: float | None = None
        self.context_menu_canvas_y: float | None = None
        self.is_spacebar_held: bool = False
        self.pan_button: str | None = None
        self.ctrl_override_active: bool = False
        self.ctrl_override_previous_tool: str | None = None
        self.file_search_var = tk.StringVar(value="")
        self.applied_search_query: str = ""
        self.file_sort_var = tk.StringVar(value="Name (Natural A-Z)")
        self.show_bounding_boxes_var = tk.BooleanVar(value=True)
        self.show_labels_var = tk.BooleanVar(value=True)
        self.bounding_box_thickness_var = tk.IntVar(value=2)

    def _configure_layout(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=320)
        self.grid_columnconfigure(1, weight=1)

    def _create_main_frames(self) -> None:
        self.left_panel = ctk.CTkFrame(self, width=320, corner_radius=12)
        self.left_panel.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        self.left_panel.grid_rowconfigure(3, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)

        self.right_panel = ctk.CTkFrame(self, corner_radius=12)
        self.right_panel.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        self.right_panel.grid_rowconfigure(2, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

    def _build_left_panel(self) -> None:
        self.left_title_label = ctk.CTkLabel(
            self.left_panel,
            text="Dataset Browser",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.left_title_label.grid(row=0, column=0, padx=16, pady=(16, 12), sticky="w")

        self.folder_controls_frame = ctk.CTkFrame(self.left_panel)
        self.folder_controls_frame.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.folder_controls_frame.grid_columnconfigure(0, weight=1)
        self.folder_controls_frame.grid_columnconfigure(1, weight=0)

        self.select_folder_button = ctk.CTkButton(
            self.folder_controls_frame,
            text="Select Dataset Folder",
            command=self.load_dataset_folder,
        )
        self.select_folder_button.grid(row=0, column=0, padx=12, pady=(12, 8), sticky="ew")

        self.dataset_path_label = ctk.CTkLabel(
            self.folder_controls_frame,
            text="No dataset selected.",
            anchor="w",
            justify="left",
            wraplength=260,
            text_color=("gray40", "gray70"),
        )
        self.dataset_path_label.grid(row=1, column=0, padx=12, pady=(0, 6), sticky="ew")

        self.dataset_summary_label = ctk.CTkLabel(
            self.folder_controls_frame,
            text="Select a folder to list images and matching YOLO labels.",
            anchor="w",
            justify="left",
            wraplength=260,
            text_color=("gray40", "gray70"),
        )
        self.dataset_summary_label.grid(row=2, column=0, padx=12, pady=(0, 6), sticky="ew")

        self.selected_image_label = ctk.CTkLabel(
            self.folder_controls_frame,
            text="Selected image: None",
            anchor="w",
            justify="left",
            wraplength=260,
            text_color=("gray40", "gray70"),
        )
        self.selected_image_label.grid(row=3, column=0, padx=12, pady=(0, 12), sticky="ew")

        self.file_search_entry = ctk.CTkEntry(
            self.folder_controls_frame,
            textvariable=self.file_search_var,
            placeholder_text="Search files...",
        )
        self.file_search_entry.grid(row=4, column=0, padx=(12, 8), pady=(0, 8), sticky="ew")
        self.file_search_entry.bind("<Return>", self._on_search_entry_return)
        self.file_search_entry.bind("<KP_Enter>", self._on_search_entry_return)

        self.search_button = ctk.CTkButton(
            self.folder_controls_frame,
            text="Search",
            width=84,
            command=self._run_image_search,
        )
        self.search_button.grid(row=4, column=1, padx=(0, 12), pady=(0, 8), sticky="e")

        self.file_sort_option_menu = ctk.CTkOptionMenu(
            self.folder_controls_frame,
            variable=self.file_sort_var,
            values=[
                "Name (Natural A-Z)",
                "Name (Natural Z-A)",
                "Name (A-Z)",
                "Name (Z-A)",
                "Date Added (Newest)",
                "Date Added (Oldest)",
                "Date Modified (Newest)",
                "Date Modified (Oldest)",
                "Size (Largest)",
                "Size (Smallest)",
            ],
            command=self._on_image_sort_changed,
        )
        self.file_sort_option_menu.grid(row=5, column=0, columnspan=2, padx=12, pady=(0, 12), sticky="ew")

        self.image_list_header_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.image_list_header_frame.grid(row=2, column=0, padx=16, pady=(0, 8), sticky="ew")
        self.image_list_header_frame.grid_columnconfigure(0, weight=1)

        self.image_list_label = ctk.CTkLabel(
            self.image_list_header_frame,
            text="Image List",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        )
        self.image_list_label.grid(row=0, column=0, sticky="w")

        self.image_count_label = ctk.CTkLabel(
            self.image_list_header_frame,
            text="0 images",
            text_color=("gray40", "gray70"),
        )
        self.image_count_label.grid(row=0, column=1, sticky="e")

        self._configure_file_tree_style()
        self.image_list_container = ctk.CTkFrame(self.left_panel, corner_radius=10)
        self.image_list_container.grid(row=3, column=0, padx=16, pady=(0, 16), sticky="nsew")
        self.image_list_container.grid_rowconfigure(1, weight=1)
        self.image_list_container.grid_columnconfigure(0, weight=1)

        self.image_list_status_label = ctk.CTkLabel(
            self.image_list_container,
            text="",
            justify="left",
            anchor="w",
            text_color=("gray40", "gray70"),
        )
        self.image_list_status_label.grid(row=0, column=0, columnspan=2, padx=8, pady=(8, 4), sticky="ew")
        self.image_list_status_label.grid_remove()

        self.image_tree = ttk.Treeview(
            self.image_list_container,
            columns=("label", "details"),
            show="tree headings",
            selectmode="browse",
            style="Datomate.Treeview",
        )
        self.image_tree.heading("#0", text="File")
        self.image_tree.heading("label", text="Lbl")
        self.image_tree.heading("details", text="Size | Added")
        self.image_tree.column("#0", width=170, stretch=True, anchor="w")
        self.image_tree.column("label", width=38, stretch=False, anchor="center")
        self.image_tree.column("details", width=150, stretch=False, anchor="w")
        self.image_tree.grid(row=1, column=0, padx=(8, 0), pady=(0, 8), sticky="nsew")
        self.image_tree.bind("<<TreeviewSelect>>", self._on_image_tree_selection)

        self.image_tree_scrollbar = ttk.Scrollbar(
            self.image_list_container,
            orient="vertical",
            command=self.image_tree.yview,
        )
        self.image_tree_scrollbar.grid(row=1, column=1, padx=(0, 8), pady=(0, 8), sticky="ns")
        self.image_tree.configure(yscrollcommand=self.image_tree_scrollbar.set)

        self._populate_image_list()

    def _create_empty_drag_data(self) -> dict[str, object]:
        return {
            "start_x": 0.0,
            "start_y": 0.0,
            "selected_object_id": None,
            "annotation_id": None,
            "handle": None,
            "start_coords": None,
            "start_coords_map": {},
            "selected_annotation_ids": set(),
            "additive": False,
        }

    def _create_empty_pan_data(self) -> dict[str, float]:
        return {
            "start_x": 0.0,
            "start_y": 0.0,
            "pan_offset_x": 0.0,
            "pan_offset_y": 0.0,
        }

    def _on_app_close(self) -> None:
        self._cancel_scheduled_canvas_render()
        self._cancel_high_quality_canvas_render()
        self._stop_list_loading_animation()
        self._stop_canvas_loading_animation()
        self._cancel_image_list_refresh()
        self._cancel_pending_image_list_render()
        if getattr(self, "background_result_after_id", None) is not None:
            self.after_cancel(self.background_result_after_id)
            self.background_result_after_id = None
        self.background_executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    def _focused_widget_allows_annotation_shortcuts(self) -> bool:
        focused_widget = self.focus_get()
        if focused_widget is None:
            return True

        return focused_widget.winfo_class() not in {"Entry", "Text", "TEntry", "Spinbox"}

    def _widget_supports_text_input(self, widget: tk.Misc | None) -> bool:
        if widget is None:
            return False

        current_widget: tk.Misc | None = widget
        while current_widget is not None:
            if current_widget.winfo_class() in {"Entry", "Text", "TEntry", "Spinbox"}:
                return True
            current_widget = current_widget.master
        return False

    def _clear_text_entry_focus(self) -> None:
        if hasattr(self, "canvas") and self.canvas.winfo_exists():
            self.canvas.focus_set()
        else:
            self.focus_set()

    def _configure_file_tree_style(self) -> None:
        appearance_mode = ctk.get_appearance_mode().lower()
        is_dark = appearance_mode == "dark"

        background = "#15171a" if is_dark else "#f5f7fb"
        field_background = "#111318" if is_dark else "#ffffff"
        foreground = "#f3f4f6" if is_dark else "#111827"
        muted_foreground = "#c7d2fe" if is_dark else "#1f2937"

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Datomate.Treeview",
            background=field_background,
            fieldbackground=field_background,
            foreground=foreground,
            rowheight=26,
            borderwidth=0,
            relief="flat",
        )
        style.configure(
            "Datomate.Treeview.Heading",
            background=background,
            foreground=muted_foreground,
            relief="flat",
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Datomate.Treeview",
            background=[("selected", "#1f6aa5")],
            foreground=[("selected", "#ffffff")],
        )

    def _on_global_left_click(self, event: tk.Event) -> None:
        if not self._widget_supports_text_input(self.focus_get()):
            return
        if self._widget_supports_text_input(event.widget):
            return
        self.after_idle(self._clear_text_entry_focus)

    def _submit_background_task(
        self,
        task_kind: str,
        token: int,
        task_data: dict[str, object],
        function: object,
        *args: object,
    ) -> None:
        def worker() -> None:
            try:
                result = function(*args)
                self.background_result_queue.put((task_kind, token, task_data, None, result))
            except Exception as exc:
                self.background_result_queue.put((task_kind, token, task_data, exc, None))

        self.background_executor.submit(worker)

    def _poll_background_results(self) -> None:
        try:
            while True:
                task_kind, token, task_data, error, result = self.background_result_queue.get_nowait()
                if task_kind == "dataset_scan":
                    self._complete_dataset_scan(token, error, result)
                elif task_kind == "image_list_refresh":
                    self._complete_image_list_refresh(token, error, result)
                elif task_kind == "image_load":
                    self._complete_image_load(token, task_data, error, result)
        except queue.Empty:
            pass

        if self.winfo_exists():
            self.background_result_after_id = self.after(50, self._poll_background_results)

    def _apply_canvas_cursor(self) -> None:
        if self.is_panning:
            self.canvas.configure(cursor="fleur")
            return

        if self.is_spacebar_held and self.current_pil_image is not None:
            self.canvas.configure(cursor="fleur")
            return

        if self.current_tool == "draw":
            self.canvas.configure(cursor="crosshair")
        else:
            self.canvas.configure(cursor="arrow")

    def _reset_interaction_state(self) -> None:
        self.is_drawing = False
        self.is_dragging = False
        self.is_selecting_region = False
        self.drag_data = self._create_empty_drag_data()
        if self.selection_rect_id is not None:
            self.canvas.delete(self.selection_rect_id)
            self.selection_rect_id = None
        if self.temp_rectangle_id is not None:
            self.canvas.delete(self.temp_rectangle_id)
            self.temp_rectangle_id = None

    def _next_annotation_id(self) -> int:
        self.annotation_id_counter += 1
        return self.annotation_id_counter

    def _tool_selector_value(self, tool_mode: str) -> str:
        return "\u25a3 Draw" if tool_mode == "draw" else "\u2316 Select"

    def _set_current_tool(self, tool_label: str) -> None:
        tool_name = tool_label.lower()
        self.current_tool = "draw" if "draw" in tool_name else "select"
        self._reset_interaction_state()

        if self.current_tool == "draw":
            self.tool_status_label.configure(text="Draw mode: drag to create a new box.")
        else:
            self.tool_status_label.configure(
                text="Select mode: click to select, drag selected boxes to move, or drag empty space to multi-select."
            )
        self._apply_canvas_cursor()

    def _on_ctrl_press(self, event: tk.Event) -> str | None:
        if not self._focused_widget_allows_annotation_shortcuts():
            return None
        if self.ctrl_override_active or self.is_drawing or self.is_dragging or self.is_selecting_region:
            return None

        self.ctrl_override_previous_tool = self.current_tool
        override_tool = "select" if self.current_tool == "draw" else "draw"
        self.ctrl_override_active = True
        self._set_current_tool(override_tool)
        if hasattr(self, "tool_selector"):
            self.tool_selector.set(self._tool_selector_value(override_tool))
        return None

    def _on_ctrl_release(self, event: tk.Event) -> str | None:
        if not self.ctrl_override_active:
            return None

        restore_tool = self.ctrl_override_previous_tool or "draw"
        self.ctrl_override_active = False
        self.ctrl_override_previous_tool = None
        self._set_current_tool(restore_tool)
        if hasattr(self, "tool_selector"):
            self.tool_selector.set(self._tool_selector_value(restore_tool))
        return None

    def _resolve_dataset_roots(self, dataset_path: Path) -> tuple[Path, Path | None]:
        if (dataset_path / "images").is_dir():
            images_root = dataset_path / "images"
            labels_root = dataset_path / "labels" if (dataset_path / "labels").is_dir() else None
            return images_root, labels_root

        if dataset_path.name.lower() == "images":
            labels_root = dataset_path.parent / "labels"
            return dataset_path, labels_root if labels_root.is_dir() else None

        labels_root = dataset_path / "labels"
        return dataset_path, labels_root if labels_root.is_dir() else None

    def _resolve_label_path(self, image_path: Path) -> Path:
        if (
            self.dataset_images_root is not None
            and self.dataset_labels_root is not None
            and image_path.is_relative_to(self.dataset_images_root)
        ):
            relative_path = image_path.relative_to(self.dataset_images_root).with_suffix(".txt")
            return self.dataset_labels_root / relative_path

        return image_path.with_suffix(".txt")

    def _scan_dataset_images(self, images_root: Path) -> list[dict[str, object]]:
        image_entries: list[dict[str, object]] = []
        image_extensions = {".jpg", ".jpeg", ".png"}

        for root, dirs, files in os.walk(images_root):
            dirs.sort()
            files.sort()
            root_path = Path(root)

            for file_name in files:
                image_path = root_path / file_name
                if image_path.suffix.lower() not in image_extensions:
                    continue

                relative_name = image_path.relative_to(images_root).as_posix()
                label_path = self._resolve_label_path(image_path)
                stat_result = image_path.stat()
                image_entries.append(
                    {
                        "display_name": relative_name,
                        "display_name_lower": relative_name.lower(),
                        "natural_name_key": tuple(self._natural_sort_key(relative_name)),
                        "image_path": image_path,
                        "label_path": label_path,
                        "has_label": label_path.is_file(),
                        "size_bytes": stat_result.st_size,
                        "size_display": self._format_file_size(stat_result.st_size),
                        "created_timestamp": stat_result.st_ctime,
                        "created_display": self._format_file_timestamp(stat_result.st_ctime),
                        "modified_timestamp": stat_result.st_mtime,
                        "modified_display": self._format_file_timestamp(stat_result.st_mtime),
                    }
                )
                image_entries[-1]["details_text"] = (
                    f"{image_entries[-1]['size_display']} | {image_entries[-1]['created_display']}"
                )

        return image_entries

    def _format_file_size(self, size_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB"]
        size_value = float(size_bytes)
        unit_index = 0
        while size_value >= 1024 and unit_index < len(units) - 1:
            size_value /= 1024
            unit_index += 1

        if unit_index == 0:
            return f"{int(size_value)} {units[unit_index]}"
        return f"{size_value:.1f} {units[unit_index]}"

    def _format_file_timestamp(self, timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")

    def _natural_sort_key(self, value: str) -> list[str | int]:
        parts = re.split(r"(\d+)", value.lower())
        return [int(part) if part.isdigit() else part for part in parts]

    def _compute_visible_image_entries(
        self,
        image_entries: list[dict[str, object]],
        query: str,
        sort_option: str,
    ) -> list[dict[str, object]]:
        visible_entries = [
            entry
            for entry in image_entries
            if not query or query in str(entry.get("display_name_lower", entry["display_name"])).lower()
        ]

        reverse_sort = False
        sort_key: str | None
        if sort_option == "Name (Natural Z-A)":
            sort_key = None
            reverse_sort = True
        elif sort_option == "Name (A-Z)":
            sort_key = "display_name"
        elif sort_option == "Name (Z-A)":
            sort_key = "display_name"
            reverse_sort = True
        elif sort_option == "Date Added (Newest)":
            sort_key = "created_timestamp"
            reverse_sort = True
        elif sort_option == "Date Added (Oldest)":
            sort_key = "created_timestamp"
        elif sort_option == "Date Modified (Newest)":
            sort_key = "modified_timestamp"
            reverse_sort = True
        elif sort_option == "Date Modified (Oldest)":
            sort_key = "modified_timestamp"
        elif sort_option == "Size (Largest)":
            sort_key = "size_bytes"
            reverse_sort = True
        elif sort_option == "Size (Smallest)":
            sort_key = "size_bytes"
        else:
            sort_key = None

        if sort_key is None:
            visible_entries.sort(
                key=lambda entry: tuple(
                    entry.get("natural_name_key", self._natural_sort_key(str(entry["display_name"])))
                ),
                reverse=reverse_sort,
            )
        elif sort_key == "display_name":
            visible_entries.sort(
                key=lambda entry: str(entry.get("display_name_lower", entry["display_name"])).lower(),
                reverse=reverse_sort,
            )
        else:
            visible_entries.sort(key=lambda entry: float(entry[sort_key]), reverse=reverse_sort)

        return visible_entries

    def _visible_image_entries(self) -> list[dict[str, object]]:
        return self._compute_visible_image_entries(
            self.image_entries,
            self.applied_search_query,
            self.file_sort_var.get(),
        )

    def _cancel_image_list_refresh(self) -> None:
        if self.image_list_refresh_after_id is not None:
            self.after_cancel(self.image_list_refresh_after_id)
            self.image_list_refresh_after_id = None

    def _cancel_pending_image_list_render(self) -> None:
        if self.image_list_render_after_id is not None:
            self.after_cancel(self.image_list_render_after_id)
            self.image_list_render_after_id = None
        self.pending_image_list_entries = []
        self.pending_image_list_index = 0

    def _show_image_list_message(self, message: str) -> None:
        self._clear_image_tree()
        self.image_list_status_label.configure(text=message)
        self.image_list_status_label.grid()

    def _hide_image_list_message(self) -> None:
        self.image_list_status_label.grid_remove()

    def _clear_image_tree(self) -> None:
        if not hasattr(self, "image_tree"):
            return
        children = self.image_tree.get_children()
        if children:
            self.image_tree.delete(*children)
        self.image_tree_item_ids.clear()

    def _animate_list_loading(self) -> None:
        if self.list_loading_label is None or not self.list_loading_label.winfo_exists():
            self.list_loading_after_id = None
            return

        frames = ["|", "/", "-", "\\"]
        frame = frames[self.list_loading_phase % len(frames)]
        self.list_loading_label.configure(text=f"{self.list_loading_message} {frame}")
        self.list_loading_phase += 1
        self.list_loading_after_id = self.after(140, self._animate_list_loading)

    def _start_list_loading_animation(self, message: str) -> None:
        self._cancel_pending_image_list_render()
        self._stop_list_loading_animation()
        self._clear_image_tree()
        self.list_loading_message = message
        self.list_loading_phase = 0
        self.image_list_status_label.configure(text=message)
        self.image_list_status_label.grid()
        self.list_loading_label = self.image_list_status_label
        self._animate_list_loading()

    def _stop_list_loading_animation(self) -> None:
        if self.list_loading_after_id is not None:
            self.after_cancel(self.list_loading_after_id)
            self.list_loading_after_id = None
        self.list_loading_label = None
        if hasattr(self, "image_list_status_label"):
            self.image_list_status_label.configure(text="")

    def _request_image_list_refresh(
        self,
        loading_message: str = "Updating image list",
        delay_ms: int = 90,
    ) -> None:
        self._cancel_image_list_refresh()
        if not self.image_entries:
            self._populate_image_list()
            return

        def start_refresh() -> None:
            self.image_list_refresh_after_id = None
            token = self.image_list_refresh_token + 1
            self.image_list_refresh_token = token
            image_entries_snapshot = list(self.image_entries)
            query = self.applied_search_query
            sort_option = self.file_sort_var.get()
            self._start_list_loading_animation(loading_message)
            self._submit_background_task(
                "image_list_refresh",
                token,
                {},
                self._compute_visible_image_entries,
                image_entries_snapshot,
                query,
                sort_option,
            )

        self.image_list_refresh_after_id = self.after(delay_ms, start_refresh)

    def _complete_image_list_refresh(
        self,
        token: int,
        error: Exception | None,
        result: object,
    ) -> None:
        if token != self.image_list_refresh_token:
            return

        if error is not None:
            self._stop_list_loading_animation()
            self._show_image_list_message("Unable to refresh the image list.")
            return

        if not isinstance(result, list):
            self._stop_list_loading_animation()
            self._show_image_list_message("Unable to refresh the image list.")
            return

        self._render_image_list_entries(result)

    def _add_image_list_row(self, row_index: int, entry: dict[str, object]) -> None:
        display_name = str(entry["display_name"])
        has_label = bool(entry["has_label"])
        details_text = str(
            entry.get(
                "details_text",
                f"{self._format_file_size(int(entry['size_bytes']))} | {self._format_file_timestamp(float(entry['created_timestamp']))}",
            )
        )
        item_id = self.image_tree.insert(
            "",
            "end",
            text=display_name,
            values=("●" if has_label else "", details_text),
        )
        self.image_tree_item_ids[display_name] = item_id

    def _render_next_image_list_chunk(self) -> None:
        chunk_size = 220
        end_index = min(
            self.pending_image_list_index + chunk_size,
            len(self.pending_image_list_entries),
        )
        for row_index in range(self.pending_image_list_index, end_index):
            self._add_image_list_row(row_index, self.pending_image_list_entries[row_index])

        self.pending_image_list_index = end_index
        self._refresh_image_button_states()
        if self.pending_image_list_index < len(self.pending_image_list_entries):
            self.image_list_render_after_id = self.after_idle(self._render_next_image_list_chunk)
        else:
            self.image_list_render_after_id = None

    def _render_image_list_entries(self, entries: list[dict[str, object]]) -> None:
        self._stop_list_loading_animation()
        self._cancel_pending_image_list_render()
        self._hide_image_list_message()
        self._clear_image_tree()
        self.filtered_image_entries = entries
        if self.applied_search_query:
            self.image_count_label.configure(
                text=f"{len(self.filtered_image_entries)} of {len(self.image_entries)} images"
            )
        else:
            self.image_count_label.configure(text=f"{len(self.filtered_image_entries)} images")

        if not self.image_entries:
            self._show_image_list_message("No images to display yet.\nChoose a dataset folder to begin.")
            return

        if not self.filtered_image_entries:
            self._show_image_list_message("No files match the current search.")
            return

        self.pending_image_list_entries = self.filtered_image_entries
        self.pending_image_list_index = 0
        self._render_next_image_list_chunk()

    def _run_image_search(self) -> None:
        self.applied_search_query = self.file_search_var.get().strip().lower()
        self._request_image_list_refresh("Searching images", delay_ms=0)

    def _on_search_entry_return(self, event: tk.Event) -> str:
        self._run_image_search()
        self._clear_text_entry_focus()
        return "break"

    def _on_image_filter_changed(self, *args: object) -> None:
        return

    def _on_image_sort_changed(self, value: str) -> None:
        self.file_sort_var.set(value)
        self._request_image_list_refresh("Sorting images")

    def _refresh_dataset_summary(self) -> None:
        if not self.image_entries:
            self.dataset_summary_label.configure(
                text="No .jpg, .jpeg, or .png images were found in the selected folder."
            )
            return

        labeled_count = sum(1 for entry in self.image_entries if bool(entry["has_label"]))
        self.dataset_summary_label.configure(
            text=(
                f"Found {len(self.image_entries)} images. "
                f"{labeled_count} have matching YOLO .txt files."
            )
        )

    def load_dataset_folder(self) -> None:
        selected_directory = ctk.filedialog.askdirectory(title="Select Dataset Folder")
        if not selected_directory:
            return

        self._cancel_scheduled_canvas_render()
        self._cancel_image_list_refresh()
        self._cancel_pending_image_list_render()
        self.dataset_path = Path(selected_directory)
        self.dataset_images_root, self.dataset_labels_root = self._resolve_dataset_roots(self.dataset_path)
        self.selected_image_filename = None
        self.current_annotations = []
        self.current_image_path = None
        self.current_label_path = None
        self.current_pil_image = None
        self.current_preview_levels = []
        self.canvas_image_ref = None
        self.selected_annotation_id = None
        self.selected_annotation_ids.clear()
        self.zoom_factor = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.annotation_clipboard.clear()
        self.file_search_var.set("")
        self.applied_search_query = ""
        self.file_sort_var.set("Name (Natural A-Z)")
        self._refresh_history_buttons()
        self.selected_image_label.configure(text="Selected image: None")
        self._reset_interaction_state()

        self.dataset_path_label.configure(text=f"Dataset: {self.dataset_path}")
        self.dataset_summary_label.configure(text="Scanning dataset for images and labels...")
        self._start_list_loading_animation("Scanning dataset")
        self._draw_canvas_message("Scanning dataset...")

        token = self.dataset_scan_token + 1
        self.dataset_scan_token = token
        self._submit_background_task(
            "dataset_scan",
            token,
            {},
            self._scan_dataset_images,
            self.dataset_images_root,
        )

    def _complete_dataset_scan(
        self,
        token: int,
        error: Exception | None,
        result: object,
    ) -> None:
        if token != self.dataset_scan_token:
            return

        if error is not None or not isinstance(result, list):
            self.image_entries = []
            self.filtered_image_entries = []
            self.image_path_lookup = {}
            self.image_filenames = []
            self._stop_list_loading_animation()
            self.dataset_summary_label.configure(text="Unable to scan the selected folder.")
            self._show_image_list_message("Unable to scan the selected folder.")
            self._draw_canvas_message("Unable to scan the selected folder.")
            return

        self.image_entries = result
        self.image_path_lookup = {
            str(entry["display_name"]): entry["image_path"] for entry in self.image_entries
        }
        self.image_filenames = [str(entry["display_name"]) for entry in self.image_entries]
        self._refresh_dataset_summary()
        self._request_image_list_refresh("Loading images", delay_ms=10)
        if self.image_entries:
            self._draw_canvas_message("Select an image from the list to display it.")
        else:
            self._draw_canvas_message("No images found in the selected folder.")

    def _populate_image_list(self) -> None:
        self._render_image_list_entries(self._visible_image_entries())

    def _indicator_color(self, has_label: bool) -> str:
        return "#22c55e" if has_label else "#6b7280"

    def _update_image_label_status(self, image_filename: str, has_label: bool) -> None:
        for entry in self.image_entries:
            if str(entry["display_name"]) != image_filename:
                continue

            entry["has_label"] = has_label
            item_id = self.image_tree_item_ids.get(image_filename)
            if item_id is not None:
                current_values = list(self.image_tree.item(item_id, "values"))
                details_value = current_values[1] if len(current_values) > 1 else str(entry.get("details_text", ""))
                self.image_tree.item(item_id, values=("●" if has_label else "", details_value))
            break

        self._refresh_dataset_summary()

    def _snapshot_current_annotations(self) -> dict[str, object]:
        return {
            "annotations": [annotation.copy() for annotation in self.current_annotations],
            "selected_annotation_id": self.selected_annotation_id,
            "selected_annotation_ids": sorted(self.selected_annotation_ids),
        }

    def _restore_annotation_snapshot(self, snapshot: dict[str, object]) -> None:
        annotations = snapshot.get("annotations", [])
        if not isinstance(annotations, list):
            return

        self.current_annotations = [
            annotation.copy() for annotation in annotations if isinstance(annotation, dict)
        ]
        valid_ids = {int(annotation["annotation_id"]) for annotation in self.current_annotations}
        selected_annotation_id = snapshot.get("selected_annotation_id")
        selected_annotation_ids = snapshot.get("selected_annotation_ids", [])
        if not isinstance(selected_annotation_ids, list):
            selected_annotation_ids = []

        restored_selection = {
            int(annotation_id)
            for annotation_id in selected_annotation_ids
            if isinstance(annotation_id, int) and annotation_id in valid_ids
        }
        self.selected_annotation_ids = restored_selection
        if (
            isinstance(selected_annotation_id, int)
            and selected_annotation_id in restored_selection
        ):
            self.selected_annotation_id = selected_annotation_id
        elif restored_selection:
            self.selected_annotation_id = next(iter(restored_selection))
        else:
            self.selected_annotation_id = None

        if valid_ids:
            self.annotation_id_counter = max(self.annotation_id_counter, max(valid_ids))

        self._save_current_annotations()
        self._refresh_history_buttons()
        self._render_canvas_scene()

    def _set_selected_annotations(
        self,
        annotation_ids: set[int],
        primary_annotation_id: int | None = None,
    ) -> None:
        valid_ids = {
            annotation_id
            for annotation_id in annotation_ids
            if self._get_annotation_by_id(annotation_id) is not None
        }
        self.selected_annotation_ids = valid_ids
        if (
            primary_annotation_id is not None
            and primary_annotation_id in valid_ids
        ):
            self.selected_annotation_id = primary_annotation_id
        elif self.selected_annotation_id in valid_ids:
            pass
        elif valid_ids:
            self.selected_annotation_id = sorted(valid_ids)[-1]
        else:
            self.selected_annotation_id = None

        self._refresh_annotation_styles()

    def _selected_annotations(self) -> list[dict[str, int | float]]:
        return [
            annotation
            for annotation in self.current_annotations
            if int(annotation["annotation_id"]) in self.selected_annotation_ids
        ]

    def _toggle_annotation_selection(self, annotation_id: int) -> None:
        updated_selection = set(self.selected_annotation_ids)
        if annotation_id in updated_selection:
            updated_selection.remove(annotation_id)
            primary_annotation_id = self.selected_annotation_id
            if primary_annotation_id == annotation_id:
                primary_annotation_id = None
        else:
            updated_selection.add(annotation_id)
            primary_annotation_id = annotation_id

        self._set_selected_annotations(updated_selection, primary_annotation_id)

    def _delete_annotations_by_ids(self, annotation_ids: set[int]) -> None:
        if not annotation_ids:
            return

        self.current_annotations = [
            annotation
            for annotation in self.current_annotations
            if int(annotation["annotation_id"]) not in annotation_ids
        ]
        self._set_selected_annotations(set())

    def _push_undo_state(self) -> None:
        if self.current_image_path is None:
            return

        self.undo_stack.append(self._snapshot_current_annotations())
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._refresh_history_buttons()

    def _undo(self) -> None:
        if not self.undo_stack or self.current_image_path is None:
            return

        self.redo_stack.append(self._snapshot_current_annotations())
        snapshot = self.undo_stack.pop()
        self._restore_annotation_snapshot(snapshot)

    def _redo(self) -> None:
        if not self.redo_stack or self.current_image_path is None:
            return

        self.undo_stack.append(self._snapshot_current_annotations())
        snapshot = self.redo_stack.pop()
        self._restore_annotation_snapshot(snapshot)

    def _refresh_history_buttons(self) -> None:
        if not hasattr(self, "undo_button") or not hasattr(self, "redo_button"):
            return

        undo_state = "normal" if self.undo_stack else "disabled"
        redo_state = "normal" if self.redo_stack else "disabled"
        self.undo_button.configure(state=undo_state)
        self.redo_button.configure(state=redo_state)

    def _on_undo_shortcut(self, event: tk.Event) -> str:
        self._undo()
        return "break"

    def _on_redo_shortcut(self, event: tk.Event) -> str:
        self._redo()
        return "break"

    def _on_copy_shortcut(self, event: tk.Event) -> str | None:
        if not self._focused_widget_allows_annotation_shortcuts():
            return None
        self._copy_selected_annotations()
        return "break"

    def _on_paste_shortcut(self, event: tk.Event) -> str | None:
        if not self._focused_widget_allows_annotation_shortcuts():
            return None
        self._paste_annotations()
        return "break"

    def _on_duplicate_shortcut(self, event: tk.Event) -> str | None:
        if not self._focused_widget_allows_annotation_shortcuts():
            return None
        self._duplicate_selected_annotations()
        return "break"

    def _on_delete_selected(self, event: tk.Event) -> str | None:
        if not self._focused_widget_allows_annotation_shortcuts():
            return None
        if not self.selected_annotation_ids or self.current_image_path is None:
            return "break"

        self._push_undo_state()
        self._delete_annotations_by_ids(set(self.selected_annotation_ids))
        self._save_current_annotations()
        self._render_canvas_scene()
        return "break"

    def _on_canvas_motion(self, event: tk.Event) -> None:
        if (
            self.is_spacebar_held
            and self.current_pil_image is not None
            and not self.is_drawing
            and not self.is_dragging
            and not self.is_selecting_region
        ):
            if not self.is_panning:
                self._begin_pan(event, "space")
            else:
                self._update_pan(event)
            return

        self._update_hover_cursor(event)

    def _on_canvas_leave(self, event: tk.Event) -> None:
        self._apply_canvas_cursor()

    def _on_spacebar_press(self, event: tk.Event) -> str | None:
        if not self._focused_widget_allows_annotation_shortcuts():
            return None
        if self.is_drawing or self.is_dragging or self.is_selecting_region:
            return None
        self.is_spacebar_held = True
        self._apply_canvas_cursor()
        return "break"

    def _on_spacebar_release(self, event: tk.Event) -> str | None:
        if not self._focused_widget_allows_annotation_shortcuts():
            return None
        self.is_spacebar_held = False
        if self.is_panning and self.pan_button == "space":
            self._end_pan()
        self._apply_canvas_cursor()
        return "break"

    def _on_escape_pressed(self, event: tk.Event) -> str | None:
        if self._widget_supports_text_input(self.focus_get()):
            self._clear_text_entry_focus()
            return "break"
        return None

    def _begin_pan(self, event: tk.Event, pan_button: str) -> None:
        if self.current_pil_image is None:
            return

        self.is_panning = True
        self.pan_button = pan_button
        self.pan_data = {
            "start_x": float(event.x),
            "start_y": float(event.y),
            "pan_offset_x": self.pan_offset_x,
            "pan_offset_y": self.pan_offset_y,
        }
        self._apply_canvas_cursor()

    def _update_pan(self, event: tk.Event) -> None:
        if not self.is_panning or self.current_pil_image is None:
            return

        target_pan_x = self.pan_data["pan_offset_x"] + (float(event.x) - self.pan_data["start_x"])
        target_pan_y = self.pan_data["pan_offset_y"] + (float(event.y) - self.pan_data["start_y"])
        clamped_pan_x, clamped_pan_y = self._clamp_pan_offsets(target_pan_x, target_pan_y)
        dx = clamped_pan_x - self.pan_offset_x
        dy = clamped_pan_y - self.pan_offset_y
        self.pan_offset_x = clamped_pan_x
        self.pan_offset_y = clamped_pan_y
        self._move_canvas_view(dx, dy)

    def _end_pan(self) -> None:
        self.is_panning = False
        self.pan_button = None
        self.pan_data = self._create_empty_pan_data()
        if self.current_pil_image is not None:
            self._schedule_high_quality_canvas_render(delay_ms=30)
        self._apply_canvas_cursor()

    def _canvas_point_to_normalized(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        clamped_x, clamped_y = self._clamp_to_image(canvas_x, canvas_y)
        offset_x = self.canvas_display_state["offset_x"]
        offset_y = self.canvas_display_state["offset_y"]
        display_width = self.canvas_display_state["display_width"]
        display_height = self.canvas_display_state["display_height"]
        return (
            min(max((clamped_x - offset_x) / display_width, 0.0), 1.0),
            min(max((clamped_y - offset_y) / display_height, 0.0), 1.0),
        )

    def _selected_group_canvas_center(self) -> tuple[float, float] | None:
        selected_annotations = self._selected_annotations()
        if not selected_annotations:
            return None

        left = float("inf")
        top = float("inf")
        right = float("-inf")
        bottom = float("-inf")
        for annotation in selected_annotations:
            x1, y1, x2, y2 = self._annotation_to_canvas_coords(annotation)
            left = min(left, x1, x2)
            top = min(top, y1, y2)
            right = max(right, x1, x2)
            bottom = max(bottom, y1, y2)

        return (left + right) / 2, (top + bottom) / 2

    def _copy_selected_annotations(self) -> bool:
        selected_annotations = self._selected_annotations()
        if not selected_annotations:
            return False

        left = min(
            float(annotation["x_center"]) - (float(annotation["width"]) / 2)
            for annotation in selected_annotations
        )
        top = min(
            float(annotation["y_center"]) - (float(annotation["height"]) / 2)
            for annotation in selected_annotations
        )
        right = max(
            float(annotation["x_center"]) + (float(annotation["width"]) / 2)
            for annotation in selected_annotations
        )
        bottom = max(
            float(annotation["y_center"]) + (float(annotation["height"]) / 2)
            for annotation in selected_annotations
        )
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2

        self.annotation_clipboard = [
            {
                "class_id": int(annotation["class_id"]),
                "x_offset": float(annotation["x_center"]) - center_x,
                "y_offset": float(annotation["y_center"]) - center_y,
                "width": float(annotation["width"]),
                "height": float(annotation["height"]),
            }
            for annotation in selected_annotations
        ]
        return True

    def _paste_annotations(self, canvas_x: float | None = None, canvas_y: float | None = None) -> None:
        if not self.annotation_clipboard or self.current_image_path is None:
            return

        if canvas_x is None or canvas_y is None:
            group_center = self._selected_group_canvas_center()
            if group_center is not None:
                canvas_x = group_center[0] + 18
                canvas_y = group_center[1] + 18
            else:
                canvas_x = self.canvas_display_state["offset_x"] + (self.canvas_display_state["display_width"] / 2)
                canvas_y = self.canvas_display_state["offset_y"] + (self.canvas_display_state["display_height"] / 2)

        target_x, target_y = self._canvas_point_to_normalized(canvas_x, canvas_y)
        self._push_undo_state()
        new_annotation_ids: set[int] = set()

        for clipboard_annotation in self.annotation_clipboard:
            width = float(clipboard_annotation["width"])
            height = float(clipboard_annotation["height"])
            x_center = target_x + float(clipboard_annotation["x_offset"])
            y_center = target_y + float(clipboard_annotation["y_offset"])
            x_center = min(max(x_center, width / 2), 1.0 - (width / 2))
            y_center = min(max(y_center, height / 2), 1.0 - (height / 2))
            annotation_id = self._next_annotation_id()
            self.current_annotations.append(
                {
                    "annotation_id": annotation_id,
                    "class_id": int(clipboard_annotation["class_id"]),
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }
            )
            new_annotation_ids.add(annotation_id)

        self._set_selected_annotations(new_annotation_ids, max(new_annotation_ids))
        self._save_current_annotations()
        self._render_canvas_scene()

    def _cut_selected_annotations(self) -> None:
        if not self._copy_selected_annotations() or not self.selected_annotation_ids:
            return

        self._push_undo_state()
        self._delete_annotations_by_ids(set(self.selected_annotation_ids))
        self._save_current_annotations()
        self._render_canvas_scene()

    def _duplicate_selected_annotations(self) -> None:
        if not self._copy_selected_annotations():
            return

        self._paste_annotations()

    def _show_canvas_context_menu(self, event: tk.Event) -> None:
        if self.current_pil_image is None:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        self.context_menu_canvas_x = canvas_x
        self.context_menu_canvas_y = canvas_y

        hit_result = self._hit_test_annotation(canvas_x, canvas_y)
        if hit_result is not None:
            annotation_id, _ = hit_result
            if annotation_id not in self.selected_annotation_ids:
                self._set_selected_annotations({annotation_id}, annotation_id)
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Copy\tCtrl+C", command=self._copy_selected_annotations)
            menu.add_command(label="Cut", command=self._cut_selected_annotations)
            menu.add_command(label="Duplicate\tCtrl+D", command=self._duplicate_selected_annotations)
            menu.add_command(
                label="Paste\tCtrl+V",
                state=tk.NORMAL if self.annotation_clipboard else tk.DISABLED,
                command=lambda: self._paste_annotations(self.context_menu_canvas_x, self.context_menu_canvas_y),
            )
            menu.add_command(
                label="Delete\tDel / Backspace",
                command=lambda: self._on_delete_selected(event),
            )
        else:
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(
                label="Paste\tCtrl+V",
                state=tk.NORMAL if self.annotation_clipboard else tk.DISABLED,
                command=lambda: self._paste_annotations(self.context_menu_canvas_x, self.context_menu_canvas_y),
            )

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _show_view_menu(self) -> None:
        menu = tk.Menu(self, tearoff=0)
        menu.add_checkbutton(
            label="Show bounding box",
            variable=self.show_bounding_boxes_var,
            command=self._on_view_option_changed,
        )
        menu.add_checkbutton(
            label="Show label",
            variable=self.show_labels_var,
            command=self._on_view_option_changed,
        )

        thickness_menu = tk.Menu(menu, tearoff=0)
        for thickness in range(1, 6):
            thickness_menu.add_radiobutton(
                label=f"{thickness}px",
                value=thickness,
                variable=self.bounding_box_thickness_var,
                command=self._on_view_option_changed,
            )
        menu.add_cascade(label="Bounding box thickness", menu=thickness_menu)

        try:
            menu.tk_popup(
                self.view_button.winfo_rootx(),
                self.view_button.winfo_rooty() + self.view_button.winfo_height(),
            )
        finally:
            menu.grab_release()

    def _on_view_option_changed(self) -> None:
        if self.current_pil_image is None:
            return
        self._refresh_annotation_styles()

    def _on_image_tree_selection(self, event: tk.Event) -> None:
        if self.image_tree_selection_guard:
            return

        selection = self.image_tree.selection()
        if not selection:
            return

        image_filename = str(self.image_tree.item(selection[0], "text"))
        if image_filename and image_filename != self.selected_image_filename:
            self._select_image(image_filename)

    def _select_image(self, image_filename: str) -> None:
        self.display_image(image_filename)

    def _refresh_image_button_states(self) -> None:
        if not hasattr(self, "image_tree"):
            return

        self.image_tree_selection_guard = True
        try:
            current_item_id = (
                self.image_tree_item_ids.get(self.selected_image_filename)
                if self.selected_image_filename is not None
                else None
            )
            if current_item_id is None:
                self.image_tree.selection_remove(self.image_tree.selection())
                return

            self.image_tree.selection_set(current_item_id)
            self.image_tree.focus(current_item_id)
            self.image_tree.see(current_item_id)
        finally:
            self.image_tree_selection_guard = False

    def _draw_canvas_message(self, message: str) -> None:
        self._cancel_scheduled_canvas_render()
        self._cancel_high_quality_canvas_render()
        self._stop_canvas_loading_animation()
        self.canvas.delete("all")
        self.canvas_image_ref = None
        self.canvas_image_item_id = None
        self.canvas_image_render_bounds = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
        self.annotation_canvas_items.clear()
        self.canvas_item_to_annotation_id.clear()
        self.selection_handle_items.clear()
        self.handle_item_to_name.clear()
        self.temp_rectangle_id = None
        self.update_idletasks()

        canvas_width = max(self.canvas.winfo_width(), self.canvas.winfo_reqwidth(), 1)
        canvas_height = max(self.canvas.winfo_height(), self.canvas.winfo_reqheight(), 1)

        self.canvas.create_text(
            canvas_width / 2,
            canvas_height / 2,
            text=message,
            fill="#d1d5db",
            font=("Segoe UI", 12),
            justify="center",
        )

    def _animate_canvas_loading(self) -> None:
        if self.canvas_loading_text_id is None or not self.canvas.winfo_exists():
            self.canvas_loading_after_id = None
            return

        frames = ["|", "/", "-", "\\"]
        frame = frames[self.canvas_loading_phase % len(frames)]
        self.canvas.itemconfigure(
            self.canvas_loading_text_id,
            text=f"{self.canvas_loading_message} {frame}",
        )
        self.canvas_loading_phase += 1
        self.canvas_loading_after_id = self.after(140, self._animate_canvas_loading)

    def _show_canvas_loading(self, message: str) -> None:
        self._cancel_scheduled_canvas_render()
        self._cancel_high_quality_canvas_render()
        self._stop_canvas_loading_animation()
        self.canvas.delete("all")
        self.canvas_image_ref = None
        self.canvas_image_item_id = None
        self.canvas_image_render_bounds = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
        self.annotation_canvas_items.clear()
        self.canvas_item_to_annotation_id.clear()
        self.selection_handle_items.clear()
        self.handle_item_to_name.clear()
        self.temp_rectangle_id = None
        self.update_idletasks()

        canvas_width = max(self.canvas.winfo_width(), self.canvas.winfo_reqwidth(), 1)
        canvas_height = max(self.canvas.winfo_height(), self.canvas.winfo_reqheight(), 1)
        self.canvas_loading_message = message
        self.canvas_loading_phase = 0
        self.canvas_loading_text_id = self.canvas.create_text(
            canvas_width / 2,
            canvas_height / 2,
            text=message,
            fill="#d1d5db",
            font=("Segoe UI", 12),
            justify="center",
        )
        self._animate_canvas_loading()

    def _stop_canvas_loading_animation(self) -> None:
        if self.canvas_loading_after_id is not None:
            self.after_cancel(self.canvas_loading_after_id)
            self.canvas_loading_after_id = None
        self.canvas_loading_text_id = None

    def _cancel_scheduled_canvas_render(self) -> None:
        if self.pending_render_after_id is not None:
            self.after_cancel(self.pending_render_after_id)
            self.pending_render_after_id = None
        self.pending_force_rebuild = False
        self.pending_interactive_render = False

    def _cancel_high_quality_canvas_render(self) -> None:
        if self.high_quality_render_after_id is not None:
            self.after_cancel(self.high_quality_render_after_id)
            self.high_quality_render_after_id = None

    def _schedule_canvas_render(self, force_rebuild: bool = False, interactive: bool = False) -> None:
        if force_rebuild:
            self.pending_force_rebuild = True

        if self.pending_render_after_id is None:
            self.pending_interactive_render = interactive
            self.pending_render_after_id = self.after(16, self._flush_scheduled_canvas_render)
            return

        if not interactive:
            self.pending_interactive_render = False

    def _flush_scheduled_canvas_render(self) -> None:
        self.pending_render_after_id = None
        force_rebuild = self.pending_force_rebuild
        interactive = self.pending_interactive_render
        self.pending_force_rebuild = False
        self.pending_interactive_render = False
        self._render_canvas_scene(force_rebuild=force_rebuild, interactive=interactive)

    def _schedule_high_quality_canvas_render(self, delay_ms: int = 120) -> None:
        self._cancel_high_quality_canvas_render()
        self.high_quality_render_after_id = self.after(delay_ms, self._render_high_quality_canvas_scene)

    def _render_high_quality_canvas_scene(self) -> None:
        self.high_quality_render_after_id = None
        if self.current_pil_image is None:
            return
        self._render_canvas_scene(refresh_image=True)

    def _load_image_bundle(
        self,
        image_path: Path,
        label_path: Path,
    ) -> tuple[Image.Image, list[tuple[int, Image.Image]], list[dict[str, int | float]]]:
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB").copy()
        resampling = getattr(Image, "Resampling", Image)
        preview_levels: list[tuple[int, Image.Image]] = []
        original_max_dimension = max(image.size)
        for max_dimension in (256, 512, 768, 1024, 1600):
            if max_dimension >= original_max_dimension:
                continue
            preview_image = image.copy()
            preview_image.thumbnail((max_dimension, max_dimension), resampling.BILINEAR)
            preview_levels.append((max(preview_image.size), preview_image))

        annotations: list[dict[str, int | float]] = []
        if label_path.is_file():
            with label_path.open("r", encoding="utf-8") as label_file:
                for raw_line in label_file:
                    line = raw_line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        continue

                    try:
                        annotations.append(
                            {
                                "class_id": int(float(parts[0])),
                                "x_center": float(parts[1]),
                                "y_center": float(parts[2]),
                                "width": float(parts[3]),
                                "height": float(parts[4]),
                            }
                        )
                    except ValueError:
                        continue

        return image, preview_levels, annotations

    def _select_render_source_image(
        self,
        display_width: int,
        display_height: int,
        interactive: bool,
    ) -> Image.Image:
        if self.current_pil_image is None:
            raise ValueError("No image is loaded.")

        target_max_dimension = max(display_width, display_height)
        if target_max_dimension <= 0 or not self.current_preview_levels:
            return self.current_pil_image

        required_source_dimension = target_max_dimension * (1.0 if interactive else 1.2)
        for source_dimension, preview_image in self.current_preview_levels:
            if source_dimension >= required_source_dimension:
                return preview_image

        largest_preview_dimension, largest_preview_image = self.current_preview_levels[-1]
        if required_source_dimension <= largest_preview_dimension * 1.15:
            return largest_preview_image
        return self.current_pil_image

    def _visible_image_region(
        self,
        canvas_width: int,
        canvas_height: int,
        offset_x: float,
        offset_y: float,
        display_width: int,
        display_height: int,
        include_overscan: bool = False,
    ) -> tuple[float, float, float, float] | None:
        image_left = offset_x
        image_top = offset_y
        image_right = offset_x + display_width
        image_bottom = offset_y + display_height

        visible_left = max(image_left, 0.0)
        visible_top = max(image_top, 0.0)
        visible_right = min(image_right, float(canvas_width))
        visible_bottom = min(image_bottom, float(canvas_height))

        if visible_right <= visible_left or visible_bottom <= visible_top:
            return None

        if not include_overscan:
            return visible_left, visible_top, visible_right, visible_bottom

        overscan_x = min(float(canvas_width) * 0.2, 180.0)
        overscan_y = min(float(canvas_height) * 0.2, 180.0)
        return (
            max(image_left, visible_left - overscan_x),
            max(image_top, visible_top - overscan_y),
            min(image_right, visible_right + overscan_x),
            min(image_bottom, visible_bottom + overscan_y),
        )

    def _render_bounds_cover_visible_region(
        self,
        canvas_width: int,
        canvas_height: int,
        offset_x: float,
        offset_y: float,
        display_width: int,
        display_height: int,
    ) -> bool:
        visible_region = self._visible_image_region(
            canvas_width,
            canvas_height,
            offset_x,
            offset_y,
            display_width,
            display_height,
            include_overscan=False,
        )
        if visible_region is None:
            return False

        return (
            self.canvas_image_render_bounds["left"] <= visible_region[0]
            and self.canvas_image_render_bounds["top"] <= visible_region[1]
            and self.canvas_image_render_bounds["right"] >= visible_region[2]
            and self.canvas_image_render_bounds["bottom"] >= visible_region[3]
        )

    def _load_yolo_annotations(self, label_path: Path) -> list[dict[str, int | float]]:
        if not label_path.is_file():
            return []

        annotations: list[dict[str, int | float]] = []
        with label_path.open("r", encoding="utf-8") as label_file:
            for raw_line in label_file:
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    continue

                try:
                    class_id = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    continue

                annotations.append(
                    {
                        "annotation_id": self._next_annotation_id(),
                        "class_id": class_id,
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                    }
                )

        return annotations

    def _get_annotation_color(self, class_id: int) -> str:
        return self.annotation_colors[class_id % len(self.annotation_colors)]

    def _annotation_label_text(self, annotation: dict[str, int | float]) -> str:
        class_id = int(annotation["class_id"])
        return str(class_id)

    def _get_annotation_by_id(self, annotation_id: int) -> dict[str, int | float] | None:
        for annotation in self.current_annotations:
            if int(annotation["annotation_id"]) == annotation_id:
                return annotation
        return None

    def _annotation_to_canvas_coords(
        self,
        annotation: dict[str, int | float],
    ) -> tuple[float, float, float, float]:
        offset_x = self.canvas_display_state["offset_x"]
        offset_y = self.canvas_display_state["offset_y"]
        display_width = self.canvas_display_state["display_width"]
        display_height = self.canvas_display_state["display_height"]

        x_center_px = offset_x + (float(annotation["x_center"]) * display_width)
        y_center_px = offset_y + (float(annotation["y_center"]) * display_height)
        box_width_px = float(annotation["width"]) * display_width
        box_height_px = float(annotation["height"]) * display_height

        x1 = x_center_px - (box_width_px / 2)
        y1 = y_center_px - (box_height_px / 2)
        x2 = x_center_px + (box_width_px / 2)
        y2 = y_center_px + (box_height_px / 2)
        return x1, y1, x2, y2

    def _canvas_coords_to_normalized(
        self,
        coords: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float] | None:
        x1, y1, x2, y2 = coords
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)

        offset_x = self.canvas_display_state["offset_x"]
        offset_y = self.canvas_display_state["offset_y"]
        display_width = self.canvas_display_state["display_width"]
        display_height = self.canvas_display_state["display_height"]
        image_right = offset_x + display_width
        image_bottom = offset_y + display_height

        left = min(max(left, offset_x), image_right)
        top = min(max(top, offset_y), image_bottom)
        right = min(max(right, offset_x), image_right)
        bottom = min(max(bottom, offset_y), image_bottom)

        if (right - left) < self.minimum_box_size or (bottom - top) < self.minimum_box_size:
            return None

        x_center = ((left + right) / 2 - offset_x) / display_width
        y_center = ((top + bottom) / 2 - offset_y) / display_height
        width = (right - left) / display_width
        height = (bottom - top) / display_height

        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width = min(max(width, 0.0), 1.0)
        height = min(max(height, 0.0), 1.0)
        return x_center, y_center, width, height

    def _store_canvas_item_mapping(
        self,
        annotation_id: int,
        rect_id: int,
        text_id: int,
        background_id: int,
    ) -> None:
        self.annotation_canvas_items[annotation_id] = {
            "rect_id": rect_id,
            "text_id": text_id,
            "background_id": background_id,
        }
        self.canvas_item_to_annotation_id[rect_id] = annotation_id
        self.canvas_item_to_annotation_id[text_id] = annotation_id
        self.canvas_item_to_annotation_id[background_id] = annotation_id

    def _delete_annotation_canvas_items(self, annotation_id: int) -> None:
        canvas_items = self.annotation_canvas_items.pop(annotation_id, None)
        if canvas_items is None:
            return

        for item_id in canvas_items.values():
            self.canvas_item_to_annotation_id.pop(item_id, None)
            self.canvas.delete(item_id)

    def _clear_selection_handles(self) -> None:
        for item_id in self.selection_handle_items.values():
            self.canvas.delete(item_id)

        self.selection_handle_items.clear()
        self.handle_item_to_name.clear()

    def _selection_handle_positions(
        self,
        coords: tuple[float, float, float, float],
    ) -> dict[str, tuple[float, float]]:
        x1, y1, x2, y2 = coords
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        return {
            "nw": (left, top),
            "n": (center_x, top),
            "ne": (right, top),
            "e": (right, center_y),
            "se": (right, bottom),
            "s": (center_x, bottom),
            "sw": (left, bottom),
            "w": (left, center_y),
        }

    def _update_selection_handles(self) -> None:
        if not self.selected_annotation_ids or not self.show_bounding_boxes_var.get():
            self._clear_selection_handles()
            return

        self._clear_selection_handles()
        for annotation_id in sorted(self.selected_annotation_ids):
            canvas_items = self.annotation_canvas_items.get(annotation_id)
            annotation = self._get_annotation_by_id(annotation_id)
            if canvas_items is None or annotation is None:
                continue

            color = self._get_annotation_color(int(annotation["class_id"]))
            handle_positions = self._selection_handle_positions(
                tuple(self.canvas.coords(canvas_items["rect_id"]))
            )
            for handle_name, (handle_x, handle_y) in handle_positions.items():
                oval_coords = (
                    handle_x - self.handle_radius,
                    handle_y - self.handle_radius,
                    handle_x + self.handle_radius,
                    handle_y + self.handle_radius,
                )
                handle_id = self.canvas.create_oval(
                    *oval_coords,
                    fill="white",
                    outline=color,
                    width=2,
                    tags=("selection_handle",),
                )
                handle_key = (annotation_id, handle_name)
                self.selection_handle_items[handle_key] = handle_id
                self.handle_item_to_name[handle_id] = handle_key
                self.canvas.tag_raise(handle_id)

    def _handle_at_position(self, x: float, y: float) -> tuple[int, str] | None:
        for handle_key, handle_id in self.selection_handle_items.items():
            hx1, hy1, hx2, hy2 = self.canvas.coords(handle_id)
            if hx1 <= x <= hx2 and hy1 <= y <= hy2:
                return handle_key
        return None

    def _cursor_for_handle(self, handle: str | None) -> str:
        if handle == "move":
            return "fleur"
        if handle in {"nw", "se"}:
            return "size_nw_se"
        if handle in {"ne", "sw"}:
            return "size_ne_sw"
        if handle in {"n", "s"}:
            return "sb_v_double_arrow"
        if handle in {"e", "w"}:
            return "sb_h_double_arrow"
        return "arrow"

    def _update_hover_cursor(self, event: tk.Event | None = None) -> None:
        if self.is_panning:
            self.canvas.configure(cursor="fleur")
            return

        if self.is_drawing:
            self.canvas.configure(cursor="crosshair")
            return

        if self.is_selecting_region:
            self.canvas.configure(cursor="crosshair")
            return

        if self.is_dragging and isinstance(self.drag_data.get("handle"), str):
            self.canvas.configure(cursor=self._cursor_for_handle(self.drag_data["handle"]))
            return

        if event is None or self.current_tool != "select":
            self._apply_canvas_cursor()
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        handle_result = self._handle_at_position(canvas_x, canvas_y)
        if handle_result is not None:
            _, handle_name = handle_result
            self.canvas.configure(cursor=self._cursor_for_handle(handle_name))
            return

        current_items = self.canvas.find_withtag("current")
        if current_items:
            current_item_id = current_items[-1]
            handle_result = self.handle_item_to_name.get(current_item_id)
            if handle_result is not None:
                _, handle_name = handle_result
                self.canvas.configure(cursor=self._cursor_for_handle(handle_name))
                return

        hit_result = self._hit_test_annotation(canvas_x, canvas_y)
        if hit_result is None:
            self._apply_canvas_cursor()
            return

        _, handle = hit_result
        self.canvas.configure(cursor=self._cursor_for_handle(handle))

    def _position_annotation_label(self, annotation_id: int) -> None:
        annotation = self._get_annotation_by_id(annotation_id)
        canvas_items = self.annotation_canvas_items.get(annotation_id)
        if annotation is None or canvas_items is None:
            return

        rect_id = canvas_items["rect_id"]
        text_id = canvas_items["text_id"]
        background_id = canvas_items["background_id"]

        self.canvas.itemconfigure(text_id, text=self._annotation_label_text(annotation))

        x1, y1, x2, y2 = self.canvas.coords(rect_id)
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)

        image_left = self.canvas_display_state["offset_x"]
        image_top = self.canvas_display_state["offset_y"]
        image_right = image_left + self.canvas_display_state["display_width"]
        image_bottom = image_top + self.canvas_display_state["display_height"]

        preferred_x = left + 4
        preferred_y = top - 24
        if preferred_y < image_top:
            preferred_y = min(top + 4, image_bottom - 18)

        self.canvas.coords(text_id, preferred_x, preferred_y)
        text_bbox = self.canvas.bbox(text_id)
        if text_bbox is None:
            return

        adjust_x = 0.0
        adjust_y = 0.0
        if text_bbox[0] < image_left:
            adjust_x = image_left - text_bbox[0] + 4
        elif text_bbox[2] > image_right:
            adjust_x = image_right - text_bbox[2] - 4

        if text_bbox[1] < image_top:
            adjust_y = image_top - text_bbox[1] + 4
        elif text_bbox[3] > image_bottom:
            adjust_y = image_bottom - text_bbox[3] - 4

        if adjust_x or adjust_y:
            self.canvas.move(text_id, adjust_x, adjust_y)
            text_bbox = self.canvas.bbox(text_id)

        if text_bbox is None:
            return

        self.canvas.coords(
            background_id,
            text_bbox[0] - 4,
            text_bbox[1] - 2,
            text_bbox[2] + 4,
            text_bbox[3] + 2,
        )
        self.canvas.tag_raise(background_id, rect_id)
        self.canvas.tag_raise(text_id, background_id)

    def _set_annotation_canvas_coords(
        self,
        annotation_id: int,
        coords: tuple[float, float, float, float],
        update_handles: bool = True,
    ) -> None:
        canvas_items = self.annotation_canvas_items.get(annotation_id)
        if canvas_items is None:
            return

        self.canvas.coords(canvas_items["rect_id"], *coords)
        self._position_annotation_label(annotation_id)
        if update_handles and annotation_id in self.selected_annotation_ids:
            self._update_selection_handles()

    def _refresh_annotation_styles(self) -> None:
        show_boxes = self.show_bounding_boxes_var.get()
        show_labels = self.show_labels_var.get()
        base_thickness = max(1, int(self.bounding_box_thickness_var.get()))
        for annotation in self.current_annotations:
            annotation_id = int(annotation["annotation_id"])
            canvas_items = self.annotation_canvas_items.get(annotation_id)
            if canvas_items is None:
                continue

            rect_id = canvas_items["rect_id"]
            text_id = canvas_items["text_id"]
            background_id = canvas_items["background_id"]
            base_color = self._get_annotation_color(int(annotation["class_id"]))
            if annotation_id in self.selected_annotation_ids:
                self.canvas.itemconfigure(
                    rect_id,
                    width=base_thickness + 1,
                    outline=base_color,
                    state=tk.NORMAL if show_boxes else tk.HIDDEN,
                )
            else:
                self.canvas.itemconfigure(
                    rect_id,
                    width=base_thickness,
                    outline=base_color,
                    state=tk.NORMAL if show_boxes else tk.HIDDEN,
                )
            label_state = tk.NORMAL if show_labels else tk.HIDDEN
            self.canvas.itemconfigure(text_id, state=label_state)
            self.canvas.itemconfigure(background_id, state=label_state)

        self._update_selection_handles()

    def _render_annotation(self, annotation: dict[str, int | float]) -> None:
        annotation_id = int(annotation["annotation_id"])
        class_id = int(annotation["class_id"])
        color = self._get_annotation_color(class_id)
        x1, y1, x2, y2 = self._annotation_to_canvas_coords(annotation)

        rect_id = self.canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline=color,
            width=max(1, int(self.bounding_box_thickness_var.get())),
            tags=("annotation_box",),
        )
        text_id = self.canvas.create_text(
            x1 + 4,
            y1 + 4,
            text=self._annotation_label_text(annotation),
            anchor="nw",
            fill="white",
            font=("Segoe UI", 10, "bold"),
            tags=("annotation_label_text",),
        )
        background_id = self.canvas.create_rectangle(
            0,
            0,
            0,
            0,
            fill=color,
            outline=color,
            tags=("annotation_label_background",),
        )

        self._store_canvas_item_mapping(annotation_id, rect_id, text_id, background_id)
        self._position_annotation_label(annotation_id)

    def _sync_annotation_canvas_items(self) -> None:
        active_annotation_ids = {int(annotation["annotation_id"]) for annotation in self.current_annotations}

        for annotation_id in list(self.annotation_canvas_items):
            if annotation_id not in active_annotation_ids:
                self._delete_annotation_canvas_items(annotation_id)

        for annotation in self.current_annotations:
            annotation_id = int(annotation["annotation_id"])
            if annotation_id not in self.annotation_canvas_items:
                self._render_annotation(annotation)
                continue

            self._set_annotation_canvas_coords(
                annotation_id,
                self._annotation_to_canvas_coords(annotation),
                update_handles=False,
            )

        self._refresh_annotation_styles()

    def _render_canvas_scene(
        self,
        force_rebuild: bool = False,
        interactive: bool = False,
        refresh_image: bool = False,
    ) -> None:
        if self.current_pil_image is None:
            self._draw_canvas_message("Select an image from the list to display it.")
            return

        self._stop_canvas_loading_animation()
        self.update_idletasks()
        canvas_width = max(self.canvas.winfo_width(), self.canvas.winfo_reqwidth(), 1)
        canvas_height = max(self.canvas.winfo_height(), self.canvas.winfo_reqheight(), 1)
        image_width, image_height = self.current_pil_image.size
        previous_display_width = self.canvas_display_state["display_width"]
        previous_display_height = self.canvas_display_state["display_height"]

        base_scale = min(canvas_width / image_width, canvas_height / image_height)
        scale = base_scale * self.zoom_factor
        display_width = max(1, round(image_width * scale))
        display_height = max(1, round(image_height * scale))
        self.pan_offset_x, self.pan_offset_y = self._clamp_pan_offsets(
            self.pan_offset_x,
            self.pan_offset_y,
            display_width,
            display_height,
        )
        offset_x = ((canvas_width - display_width) / 2) + self.pan_offset_x
        offset_y = ((canvas_height - display_height) / 2) + self.pan_offset_y
        self.canvas_display_state = {
            "offset_x": offset_x,
            "offset_y": offset_y,
            "display_width": float(display_width),
            "display_height": float(display_height),
            "image_width": float(image_width),
            "image_height": float(image_height),
        }

        if force_rebuild:
            self.canvas.delete("all")
            self.canvas_image_item_id = None
            self.canvas_image_render_bounds = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
            self.annotation_canvas_items.clear()
            self.canvas_item_to_annotation_id.clear()
            self.selection_handle_items.clear()
            self.handle_item_to_name.clear()
            self.temp_rectangle_id = None

        visible_region = self._visible_image_region(
            canvas_width,
            canvas_height,
            offset_x,
            offset_y,
            display_width,
            display_height,
            include_overscan=True,
        )
        image_render_needed = (
            force_rebuild
            or refresh_image
            or self.canvas_image_item_id is None
            or display_width != previous_display_width
            or display_height != previous_display_height
            or not self._render_bounds_cover_visible_region(
                canvas_width,
                canvas_height,
                offset_x,
                offset_y,
                display_width,
                display_height,
            )
        )

        if visible_region is None:
            if self.canvas_image_item_id is not None:
                self.canvas.delete(self.canvas_image_item_id)
                self.canvas_image_item_id = None
            self.canvas_image_ref = None
            self.canvas_image_render_bounds = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
        elif image_render_needed:
            resampling = getattr(Image, "Resampling", Image)
            resample_filter = resampling.NEAREST if scale >= 1.0 else resampling.BILINEAR
            source_image = self._select_render_source_image(display_width, display_height, interactive)
            render_left, render_top, render_right, render_bottom = visible_region
            render_width = max(1, round(render_right - render_left))
            render_height = max(1, round(render_bottom - render_top))

            source_width, source_height = source_image.size
            source_left = int(((render_left - offset_x) / display_width) * source_width)
            source_top = int(((render_top - offset_y) / display_height) * source_height)
            source_right = int(((render_right - offset_x) / display_width) * source_width)
            source_bottom = int(((render_bottom - offset_y) / display_height) * source_height)

            source_left = max(0, min(source_left, source_width - 1))
            source_top = max(0, min(source_top, source_height - 1))
            source_right = max(source_left + 1, min(source_right, source_width))
            source_bottom = max(source_top + 1, min(source_bottom, source_height))

            cropped_image = source_image.crop((source_left, source_top, source_right, source_bottom))
            if cropped_image.size != (render_width, render_height):
                resized_image = cropped_image.resize((render_width, render_height), resample_filter)
            else:
                resized_image = cropped_image
            self.canvas_image_ref = ImageTk.PhotoImage(resized_image)
            self.canvas_image_render_bounds = {
                "left": render_left,
                "top": render_top,
                "right": render_left + render_width,
                "bottom": render_top + render_height,
            }

            if self.canvas_image_item_id is None:
                self.canvas_image_item_id = self.canvas.create_image(
                    render_left,
                    render_top,
                    image=self.canvas_image_ref,
                    anchor="nw",
                    tags=("image",),
                )
            else:
                self.canvas.itemconfigure(self.canvas_image_item_id, image=self.canvas_image_ref)
                self.canvas.coords(self.canvas_image_item_id, render_left, render_top)
        elif self.canvas_image_item_id is not None:
            self.canvas.coords(
                self.canvas_image_item_id,
                self.canvas_image_render_bounds["left"],
                self.canvas_image_render_bounds["top"],
            )

        if self.canvas_image_item_id is not None:
            self.canvas.tag_lower(self.canvas_image_item_id)

        self._sync_annotation_canvas_items()

    def display_image(self, image_filename: str) -> None:
        self._cancel_scheduled_canvas_render()
        self._cancel_high_quality_canvas_render()
        self._reset_interaction_state()
        self._stop_canvas_loading_animation()
        self.canvas_image_ref = None
        self.canvas_image_item_id = None
        self.canvas_image_render_bounds = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
        self.selected_annotation_id = None
        self.selected_annotation_ids.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._refresh_history_buttons()
        self.selected_image_filename = image_filename
        self.selected_image_label.configure(text=f"Selected image: {image_filename}")
        self._refresh_image_button_states()

        image_path = self.image_path_lookup.get(image_filename)
        if image_path is None or not image_path.is_file():
            self.current_image_path = None
            self.current_label_path = None
            self.current_pil_image = None
            self.current_preview_levels = []
            self.current_annotations = []
            self._draw_canvas_message("Unable to locate the selected image.")
            return

        self.current_image_path = image_path
        self.current_label_path = self._resolve_label_path(image_path)
        self.current_pil_image = None
        self.current_preview_levels = []
        self.current_annotations = []
        self._show_canvas_loading(f"Loading {image_filename}")

        token = self.image_load_token + 1
        self.image_load_token = token
        image_load_path = image_path
        label_load_path = self.current_label_path
        self._submit_background_task(
            "image_load",
            token,
            {
                "image_filename": image_filename,
                "image_path": image_load_path,
                "label_path": label_load_path,
            },
            self._load_image_bundle,
            image_load_path,
            label_load_path,
        )

    def _complete_image_load(
        self,
        token: int,
        task_data: dict[str, object],
        error: Exception | None,
        result: object,
    ) -> None:
        image_filename = str(task_data["image_filename"])
        image_path = Path(task_data["image_path"])
        label_path = Path(task_data["label_path"])
        if token != self.image_load_token or image_filename != self.selected_image_filename:
            return

        if isinstance(error, OSError):
            self.current_image_path = None
            self.current_label_path = None
            self.current_pil_image = None
            self.current_preview_levels = []
            self.current_annotations = []
            self._draw_canvas_message("Unable to open the selected image.")
            return
        if error is not None or not isinstance(result, tuple) or len(result) != 3:
            self.current_image_path = None
            self.current_label_path = None
            self.current_pil_image = None
            self.current_preview_levels = []
            self.current_annotations = []
            self._draw_canvas_message("Unable to load the selected image.")
            return

        loaded_image, preview_levels, loaded_annotations = result
        self.current_image_path = image_path
        self.current_label_path = label_path
        self.current_pil_image = loaded_image
        self.current_preview_levels = preview_levels
        self.current_annotations = [
            {
                "annotation_id": self._next_annotation_id(),
                "class_id": int(annotation["class_id"]),
                "x_center": float(annotation["x_center"]),
                "y_center": float(annotation["y_center"]),
                "width": float(annotation["width"]),
                "height": float(annotation["height"]),
            }
            for annotation in loaded_annotations
        ]
        self.zoom_factor = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self._render_canvas_scene(force_rebuild=True)

    def _save_current_annotations(self) -> None:
        if self.current_label_path is None or self.selected_image_filename is None:
            return

        lines: list[str] = []
        for annotation in self.current_annotations:
            lines.append(
                f"{int(annotation['class_id'])} "
                f"{float(annotation['x_center']):.6f} "
                f"{float(annotation['y_center']):.6f} "
                f"{float(annotation['width']):.6f} "
                f"{float(annotation['height']):.6f}"
            )

        file_contents = "\n".join(lines)
        if file_contents:
            file_contents += "\n"

        self.current_label_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_label_path.write_text(file_contents, encoding="utf-8")
        self._update_image_label_status(self.selected_image_filename, self.current_label_path.is_file())

    def _clamp_pan_offsets(
        self,
        pan_offset_x: float,
        pan_offset_y: float,
        display_width: float | None = None,
        display_height: float | None = None,
    ) -> tuple[float, float]:
        self.update_idletasks()
        canvas_width = max(self.canvas.winfo_width(), self.canvas.winfo_reqwidth(), 1)
        canvas_height = max(self.canvas.winfo_height(), self.canvas.winfo_reqheight(), 1)
        current_display_width = display_width or self.canvas_display_state["display_width"]
        current_display_height = display_height or self.canvas_display_state["display_height"]

        max_pan_x = abs((canvas_width - current_display_width) / 2)
        max_pan_y = abs((canvas_height - current_display_height) / 2)
        clamped_pan_x = min(max(pan_offset_x, -max_pan_x), max_pan_x)
        clamped_pan_y = min(max(pan_offset_y, -max_pan_y), max_pan_y)
        return clamped_pan_x, clamped_pan_y

    def _move_canvas_view(self, dx: float, dy: float) -> None:
        if dx == 0 and dy == 0:
            return

        if self.canvas_image_item_id is not None:
            self.canvas.move(self.canvas_image_item_id, dx, dy)
            self.canvas_image_render_bounds["left"] += dx
            self.canvas_image_render_bounds["right"] += dx
            self.canvas_image_render_bounds["top"] += dy
            self.canvas_image_render_bounds["bottom"] += dy

        for canvas_items in self.annotation_canvas_items.values():
            self.canvas.move(canvas_items["rect_id"], dx, dy)
            self.canvas.move(canvas_items["text_id"], dx, dy)
            self.canvas.move(canvas_items["background_id"], dx, dy)

        for handle_id in self.selection_handle_items.values():
            self.canvas.move(handle_id, dx, dy)

        if self.temp_rectangle_id is not None:
            self.canvas.move(self.temp_rectangle_id, dx, dy)

        self.canvas_display_state["offset_x"] += dx
        self.canvas_display_state["offset_y"] += dy

        if self.current_pil_image is None:
            return

        canvas_width = max(self.canvas.winfo_width(), self.canvas.winfo_reqwidth(), 1)
        canvas_height = max(self.canvas.winfo_height(), self.canvas.winfo_reqheight(), 1)
        if not self._render_bounds_cover_visible_region(
            canvas_width,
            canvas_height,
            self.canvas_display_state["offset_x"],
            self.canvas_display_state["offset_y"],
            int(self.canvas_display_state["display_width"]),
            int(self.canvas_display_state["display_height"]),
        ):
            self._schedule_canvas_render(interactive=True)

    def _coords_changed(
        self,
        before: tuple[float, float, float, float],
        after: tuple[float, float, float, float],
        tolerance: float = 0.5,
    ) -> bool:
        return any(abs(old - new) > tolerance for old, new in zip(before, after))

    def _coords_map_changed(
        self,
        start_coords_map: dict[int, tuple[float, float, float, float]],
    ) -> bool:
        for annotation_id, start_coords in start_coords_map.items():
            canvas_items = self.annotation_canvas_items.get(annotation_id)
            if canvas_items is None:
                continue

            current_coords = tuple(self.canvas.coords(canvas_items["rect_id"]))
            if self._coords_changed(start_coords, current_coords):
                return True

        return False

    def _shift_pressed(self, event: tk.Event) -> bool:
        return bool(event.state & 0x0001)

    def _selection_rect_intersects_annotation(
        self,
        selection_coords: tuple[float, float, float, float],
        annotation_id: int,
    ) -> bool:
        canvas_items = self.annotation_canvas_items.get(annotation_id)
        if canvas_items is None:
            return False

        sel_x1, sel_y1, sel_x2, sel_y2 = selection_coords
        rect_x1, rect_y1, rect_x2, rect_y2 = self.canvas.coords(canvas_items["rect_id"])

        sel_left = min(sel_x1, sel_x2)
        sel_top = min(sel_y1, sel_y2)
        sel_right = max(sel_x1, sel_x2)
        sel_bottom = max(sel_y1, sel_y2)
        rect_left = min(rect_x1, rect_x2)
        rect_top = min(rect_y1, rect_y2)
        rect_right = max(rect_x1, rect_x2)
        rect_bottom = max(rect_y1, rect_y2)

        return not (
            rect_right < sel_left
            or rect_left > sel_right
            or rect_bottom < sel_top
            or rect_top > sel_bottom
        )

    def _dragged_group_coords_map(
        self,
        x: float,
        y: float,
    ) -> dict[int, tuple[float, float, float, float]] | None:
        start_coords_map = self.drag_data["start_coords_map"]
        if not isinstance(start_coords_map, dict) or not start_coords_map:
            return None

        start_x = float(self.drag_data["start_x"])
        start_y = float(self.drag_data["start_y"])
        dx = x - start_x
        dy = y - start_y

        image_left = self.canvas_display_state["offset_x"]
        image_top = self.canvas_display_state["offset_y"]
        image_right = image_left + self.canvas_display_state["display_width"]
        image_bottom = image_top + self.canvas_display_state["display_height"]

        group_left = min(min(coords[0], coords[2]) for coords in start_coords_map.values())
        group_top = min(min(coords[1], coords[3]) for coords in start_coords_map.values())
        group_right = max(max(coords[0], coords[2]) for coords in start_coords_map.values())
        group_bottom = max(max(coords[1], coords[3]) for coords in start_coords_map.values())

        dx = min(max(dx, image_left - group_left), image_right - group_right)
        dy = min(max(dy, image_top - group_top), image_bottom - group_bottom)

        return {
            annotation_id: (
                coords[0] + dx,
                coords[1] + dy,
                coords[2] + dx,
                coords[3] + dy,
            )
            for annotation_id, coords in start_coords_map.items()
        }

    def _update_annotations_from_canvas(self, annotation_ids: set[int]) -> bool:
        updated = False
        for annotation_id in annotation_ids:
            canvas_items = self.annotation_canvas_items.get(annotation_id)
            if canvas_items is None:
                continue

            rect_coords = tuple(self.canvas.coords(canvas_items["rect_id"]))
            updated = self._update_annotation_from_canvas(annotation_id, rect_coords) or updated

        return updated

    def _on_canvas_mouse_wheel(self, event: tk.Event) -> None:
        if self.current_pil_image is None or self.is_drawing or self.is_dragging:
            return

        if hasattr(event, "delta") and event.delta:
            zoom_step = 1.1 if event.delta > 0 else 1 / 1.1
        elif getattr(event, "num", None) == 4:
            zoom_step = 1.1
        elif getattr(event, "num", None) == 5:
            zoom_step = 1 / 1.1
        else:
            return

        new_zoom = min(
            max(self.zoom_factor * zoom_step, self.min_zoom_factor),
            self.max_zoom_factor,
        )
        if abs(new_zoom - self.zoom_factor) < 1e-6:
            return

        self.update_idletasks()
        canvas_width = max(self.canvas.winfo_width(), self.canvas.winfo_reqwidth(), 1)
        canvas_height = max(self.canvas.winfo_height(), self.canvas.winfo_reqheight(), 1)
        image_width, image_height = self.current_pil_image.size
        base_scale = min(canvas_width / image_width, canvas_height / image_height)

        cursor_x = self.canvas.canvasx(event.x)
        cursor_y = self.canvas.canvasy(event.y)
        current_offset_x = self.canvas_display_state["offset_x"]
        current_offset_y = self.canvas_display_state["offset_y"]
        current_display_width = self.canvas_display_state["display_width"]
        current_display_height = self.canvas_display_state["display_height"]

        if self._point_inside_image(cursor_x, cursor_y):
            relative_x = (cursor_x - current_offset_x) / current_display_width
            relative_y = (cursor_y - current_offset_y) / current_display_height
        else:
            relative_x = 0.5
            relative_y = 0.5

        new_display_width = image_width * base_scale * new_zoom
        new_display_height = image_height * base_scale * new_zoom
        centered_offset_x = (canvas_width - new_display_width) / 2
        centered_offset_y = (canvas_height - new_display_height) / 2

        self.zoom_factor = new_zoom
        self.pan_offset_x = cursor_x - (relative_x * new_display_width) - centered_offset_x
        self.pan_offset_y = cursor_y - (relative_y * new_display_height) - centered_offset_y
        self._schedule_canvas_render(interactive=True)
        self._schedule_high_quality_canvas_render()

    def _on_middle_button_press(self, event: tk.Event) -> None:
        self._begin_pan(event, "middle")

    def _on_middle_button_drag(self, event: tk.Event) -> None:
        self._update_pan(event)

    def _on_middle_button_release(self, event: tk.Event) -> None:
        self._end_pan()

    def _point_inside_image(self, x: float, y: float) -> bool:
        image_left = self.canvas_display_state["offset_x"]
        image_top = self.canvas_display_state["offset_y"]
        image_right = image_left + self.canvas_display_state["display_width"]
        image_bottom = image_top + self.canvas_display_state["display_height"]
        return image_left <= x <= image_right and image_top <= y <= image_bottom

    def _clamp_to_image(self, x: float, y: float) -> tuple[float, float]:
        image_left = self.canvas_display_state["offset_x"]
        image_top = self.canvas_display_state["offset_y"]
        image_right = image_left + self.canvas_display_state["display_width"]
        image_bottom = image_top + self.canvas_display_state["display_height"]
        clamped_x = min(max(x, image_left), image_right)
        clamped_y = min(max(y, image_top), image_bottom)
        return clamped_x, clamped_y

    def _get_handle_for_rect(
        self,
        coords: tuple[float, float, float, float],
        x: float,
        y: float,
    ) -> str | None:
        x1, y1, x2, y2 = coords
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        if left <= x <= right and top <= y <= bottom:
            return "move"
        return None

    def _hit_test_annotation(self, x: float, y: float) -> tuple[int, str] | None:
        handle_result = self._handle_at_position(x, y)
        if handle_result is not None:
            return handle_result

        current_items = self.canvas.find_withtag("current")
        if current_items:
            current_item_id = current_items[-1]
            handle_result = self.handle_item_to_name.get(current_item_id)
            if handle_result is not None:
                return handle_result

            annotation_id = self.canvas_item_to_annotation_id.get(current_item_id)
            if annotation_id is not None:
                rect_id = self.annotation_canvas_items[annotation_id]["rect_id"]
                handle = self._get_handle_for_rect(tuple(self.canvas.coords(rect_id)), x, y)
                if handle is not None:
                    return annotation_id, handle
                return annotation_id, "move"

        for annotation in reversed(self.current_annotations):
            annotation_id = int(annotation["annotation_id"])
            canvas_items = self.annotation_canvas_items.get(annotation_id)
            if canvas_items is None:
                continue

            rect_coords = tuple(self.canvas.coords(canvas_items["rect_id"]))
            handle = self._get_handle_for_rect(rect_coords, x, y)
            if handle is not None:
                return annotation_id, handle

        return None

    def _dragged_rect_coords(self, x: float, y: float) -> tuple[float, float, float, float] | None:
        start_coords = self.drag_data["start_coords"]
        handle = self.drag_data["handle"]
        if not isinstance(start_coords, tuple) or not isinstance(handle, str):
            return None

        start_x = float(self.drag_data["start_x"])
        start_y = float(self.drag_data["start_y"])
        left, top, right, bottom = start_coords
        image_left = self.canvas_display_state["offset_x"]
        image_top = self.canvas_display_state["offset_y"]
        image_right = image_left + self.canvas_display_state["display_width"]
        image_bottom = image_top + self.canvas_display_state["display_height"]

        if handle == "move":
            dx = x - start_x
            dy = y - start_y
            dx = min(max(dx, image_left - left), image_right - right)
            dy = min(max(dy, image_top - top), image_bottom - bottom)
            return left + dx, top + dy, right + dx, bottom + dy

        clamped_x, clamped_y = self._clamp_to_image(x, y)
        new_left = left
        new_top = top
        new_right = right
        new_bottom = bottom

        if "w" in handle:
            new_left = min(max(clamped_x, image_left), right - self.minimum_box_size)
        if "e" in handle:
            new_right = max(min(clamped_x, image_right), left + self.minimum_box_size)
        if "n" in handle:
            new_top = min(max(clamped_y, image_top), bottom - self.minimum_box_size)
        if "s" in handle:
            new_bottom = max(min(clamped_y, image_bottom), top + self.minimum_box_size)

        return new_left, new_top, new_right, new_bottom

    def _update_annotation_from_canvas(
        self,
        annotation_id: int,
        coords: tuple[float, float, float, float],
    ) -> bool:
        normalized = self._canvas_coords_to_normalized(coords)
        if normalized is None:
            return False

        annotation = self._get_annotation_by_id(annotation_id)
        if annotation is None:
            return False

        x_center, y_center, width, height = normalized
        annotation["x_center"] = x_center
        annotation["y_center"] = y_center
        annotation["width"] = width
        annotation["height"] = height
        return True

    def _on_canvas_left_button_press(self, event: tk.Event) -> None:
        if self.current_pil_image is None or self.is_panning:
            return

        if self.is_spacebar_held:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        if self.current_tool == "draw":
            if not self._point_inside_image(canvas_x, canvas_y):
                return

            self._reset_interaction_state()
            self._set_selected_annotations(set())
            self.canvas.configure(cursor="crosshair")

            start_x, start_y = self._clamp_to_image(canvas_x, canvas_y)
            color = self._get_annotation_color(self.default_class_id)
            self.temp_rectangle_id = self.canvas.create_rectangle(
                start_x,
                start_y,
                start_x,
                start_y,
                outline=color,
                fill=color,
                stipple="gray25",
                width=2,
                dash=(6, 3),
                tags=("draft_annotation",),
            )
            self.is_drawing = True
            self.drag_data = {
                "start_x": start_x,
                "start_y": start_y,
                "selected_object_id": self.temp_rectangle_id,
                "annotation_id": None,
                "handle": "draw",
                "start_coords": (start_x, start_y, start_x, start_y),
            }
            return

        shift_pressed = self._shift_pressed(event)
        hit_result = self._hit_test_annotation(canvas_x, canvas_y)
        if hit_result is None:
            self._reset_interaction_state()
            original_selection = set(self.selected_annotation_ids)
            if not shift_pressed:
                self._set_selected_annotations(set())

            self.is_selecting_region = True
            self.selection_rect_id = self.canvas.create_rectangle(
                canvas_x,
                canvas_y,
                canvas_x,
                canvas_y,
                outline="#3B8ED0",
                fill="#3B8ED0",
                stipple="gray25",
                dash=(4, 2),
                width=1,
                tags=("selection_rect",),
            )
            self.drag_data = {
                "start_x": canvas_x,
                "start_y": canvas_y,
                "selected_object_id": self.selection_rect_id,
                "annotation_id": None,
                "handle": "marquee",
                "start_coords": (canvas_x, canvas_y, canvas_x, canvas_y),
                "start_coords_map": {},
                "selected_annotation_ids": original_selection,
                "additive": shift_pressed,
            }
            self.canvas.configure(cursor="crosshair")
            return

        annotation_id, handle = hit_result
        canvas_items = self.annotation_canvas_items.get(annotation_id)
        if canvas_items is None:
            return

        was_selected = annotation_id in self.selected_annotation_ids
        self._reset_interaction_state()

        if handle != "move":
            if annotation_id not in self.selected_annotation_ids:
                self._set_selected_annotations({annotation_id}, annotation_id)
                self._update_hover_cursor(event)
                return

            rect_id = canvas_items["rect_id"]
            self.is_dragging = True
            self.drag_data = {
                "start_x": canvas_x,
                "start_y": canvas_y,
                "selected_object_id": rect_id,
                "annotation_id": annotation_id,
                "handle": handle,
                "start_coords": tuple(self.canvas.coords(rect_id)),
                "start_coords_map": {},
                "selected_annotation_ids": {annotation_id},
                "additive": False,
            }
            self._set_selected_annotations(set(self.selected_annotation_ids), annotation_id)
            self._update_hover_cursor(event)
            return

        if shift_pressed:
            self._toggle_annotation_selection(annotation_id)
            self._update_hover_cursor(event)
            return

        if not was_selected:
            self._set_selected_annotations({annotation_id}, annotation_id)
            self._update_hover_cursor(event)
            return

        rect_id = canvas_items["rect_id"]
        selected_ids = set(self.selected_annotation_ids) or {annotation_id}
        start_coords_map = {
            selected_id: tuple(self.canvas.coords(self.annotation_canvas_items[selected_id]["rect_id"]))
            for selected_id in selected_ids
            if selected_id in self.annotation_canvas_items
        }
        self.is_dragging = True
        self.drag_data = {
            "start_x": canvas_x,
            "start_y": canvas_y,
            "selected_object_id": rect_id,
            "annotation_id": annotation_id,
            "handle": "move",
            "start_coords": tuple(self.canvas.coords(rect_id)),
            "start_coords_map": start_coords_map,
            "selected_annotation_ids": selected_ids,
            "additive": False,
        }
        self._set_selected_annotations(selected_ids, annotation_id)
        self._update_hover_cursor(event)

    def _on_canvas_mouse_move(self, event: tk.Event) -> None:
        if self.is_panning:
            self._update_pan(event)
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        if self.is_drawing and self.temp_rectangle_id is not None:
            end_x, end_y = self._clamp_to_image(canvas_x, canvas_y)
            start_x = float(self.drag_data["start_x"])
            start_y = float(self.drag_data["start_y"])
            self.canvas.coords(self.temp_rectangle_id, start_x, start_y, end_x, end_y)
            return

        if self.is_selecting_region and self.selection_rect_id is not None:
            start_x = float(self.drag_data["start_x"])
            start_y = float(self.drag_data["start_y"])
            self.canvas.coords(self.selection_rect_id, start_x, start_y, canvas_x, canvas_y)
            return

        if not self.is_dragging:
            self._update_hover_cursor(event)
            return

        if self.drag_data["handle"] == "move":
            updated_coords_map = self._dragged_group_coords_map(canvas_x, canvas_y)
            if updated_coords_map is None:
                return

            for annotation_id, updated_coords in updated_coords_map.items():
                self._set_annotation_canvas_coords(annotation_id, updated_coords, update_handles=False)
            self._update_selection_handles()
        else:
            annotation_id = self.drag_data["annotation_id"]
            if not isinstance(annotation_id, int):
                return

            updated_coords = self._dragged_rect_coords(canvas_x, canvas_y)
            if updated_coords is None:
                return

            self._set_annotation_canvas_coords(annotation_id, updated_coords, update_handles=False)
            self._update_selection_handles()
        self._update_hover_cursor()

    def _on_canvas_left_button_release(self, event: tk.Event) -> None:
        if self.is_panning:
            if self.pan_button == "left":
                self._end_pan()
            return

        if self.is_drawing:
            self.is_drawing = False
            if self.temp_rectangle_id is None:
                self.drag_data = self._create_empty_drag_data()
                return

            temp_coords = tuple(self.canvas.coords(self.temp_rectangle_id))
            self.canvas.delete(self.temp_rectangle_id)
            self.temp_rectangle_id = None

            normalized = self._canvas_coords_to_normalized(temp_coords)
            if normalized is None:
                self.drag_data = self._create_empty_drag_data()
                return

            x_center, y_center, width, height = normalized
            self._push_undo_state()
            annotation_id = self._next_annotation_id()
            self.current_annotations.append(
                {
                    "annotation_id": annotation_id,
                    "class_id": self.default_class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }
            )
            self._set_selected_annotations({annotation_id}, annotation_id)
            self.drag_data = self._create_empty_drag_data()
            self._save_current_annotations()
            self._render_canvas_scene()
            self._apply_canvas_cursor()
            return

        if self.is_selecting_region:
            self.is_selecting_region = False
            if self.selection_rect_id is None:
                self.drag_data = self._create_empty_drag_data()
                self._apply_canvas_cursor()
                return

            selection_coords = tuple(self.canvas.coords(self.selection_rect_id))
            self.canvas.delete(self.selection_rect_id)
            self.selection_rect_id = None

            additive = bool(self.drag_data.get("additive"))
            existing_selection = set(self.drag_data.get("selected_annotation_ids", set()))
            selection_width = abs(selection_coords[2] - selection_coords[0])
            selection_height = abs(selection_coords[3] - selection_coords[1])
            if selection_width < 4 and selection_height < 4:
                if additive:
                    self._set_selected_annotations(existing_selection, self.selected_annotation_id)
                else:
                    self._set_selected_annotations(set())
                self.drag_data = self._create_empty_drag_data()
                self._apply_canvas_cursor()
                return

            hit_ids = {
                int(annotation["annotation_id"])
                for annotation in self.current_annotations
                if self._selection_rect_intersects_annotation(
                    selection_coords,
                    int(annotation["annotation_id"]),
                )
            }
            if additive:
                new_selection = existing_selection | hit_ids
            else:
                new_selection = hit_ids

            primary_annotation_id = sorted(hit_ids)[-1] if hit_ids else self.selected_annotation_id
            self._set_selected_annotations(new_selection, primary_annotation_id)
            self.drag_data = self._create_empty_drag_data()
            self._apply_canvas_cursor()
            return

        if not self.is_dragging:
            self.drag_data = self._create_empty_drag_data()
            self._apply_canvas_cursor()
            return

        self.is_dragging = False
        if self.drag_data["handle"] == "move":
            start_coords_map = self.drag_data["start_coords_map"]
            selected_ids = set(self.drag_data["selected_annotation_ids"])
            if not isinstance(start_coords_map, dict) or not self._coords_map_changed(start_coords_map):
                self.drag_data = self._create_empty_drag_data()
                self._apply_canvas_cursor()
                return

            self._push_undo_state()
            if self._update_annotations_from_canvas(selected_ids):
                self._save_current_annotations()
            self.drag_data = self._create_empty_drag_data()
            self._render_canvas_scene()
            self._apply_canvas_cursor()
            return

        annotation_id = self.drag_data["annotation_id"]
        if not isinstance(annotation_id, int):
            self.drag_data = self._create_empty_drag_data()
            self._apply_canvas_cursor()
            return

        canvas_items = self.annotation_canvas_items.get(annotation_id)
        if canvas_items is None:
            self.drag_data = self._create_empty_drag_data()
            self._apply_canvas_cursor()
            return

        rect_coords = tuple(self.canvas.coords(canvas_items["rect_id"]))
        start_coords = self.drag_data["start_coords"]
        if isinstance(start_coords, tuple) and not self._coords_changed(start_coords, rect_coords):
            self.drag_data = self._create_empty_drag_data()
            self._apply_canvas_cursor()
            return

        self._push_undo_state()
        if self._update_annotation_from_canvas(annotation_id, rect_coords):
            self._save_current_annotations()

        self.drag_data = self._create_empty_drag_data()
        self._render_canvas_scene()
        self._apply_canvas_cursor()

    def _build_right_panel(self) -> None:
        self.canvas_title_label = ctk.CTkLabel(
            self.right_panel,
            text="Datomate - Duy, Data, Automate, Annotate",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.canvas_title_label.grid(row=0, column=0, padx=16, pady=(16, 12), sticky="w")

        self.canvas_toolbar = ctk.CTkFrame(self.right_panel)
        self.canvas_toolbar.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.canvas_toolbar.grid_columnconfigure(5, weight=1)

        self.tool_mode_label = ctk.CTkLabel(
            self.canvas_toolbar,
            text="Tool",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.tool_mode_label.grid(row=0, column=0, padx=(12, 10), pady=12, sticky="w")

        self.tool_selector = ctk.CTkSegmentedButton(
            self.canvas_toolbar,
            values=["▣ Draw", "⌖ Select"],
            command=self._set_current_tool,
        )
        self.tool_selector.grid(row=0, column=1, padx=(0, 12), pady=12, sticky="w")

        self.undo_button = ctk.CTkButton(
            self.canvas_toolbar,
            text="↶ Undo",
            width=80,
            command=self._undo,
        )
        self.undo_button.grid(row=0, column=2, padx=(0, 8), pady=12, sticky="w")

        self.redo_button = ctk.CTkButton(
            self.canvas_toolbar,
            text="↷ Redo",
            width=80,
            command=self._redo,
        )
        self.redo_button.grid(row=0, column=3, padx=(0, 12), pady=12, sticky="w")

        self.view_button = ctk.CTkButton(
            self.canvas_toolbar,
            text="◫ View",
            width=96,
            command=self._show_view_menu,
        )
        self.view_button.grid(row=0, column=4, padx=(0, 12), pady=12, sticky="w")

        self.tool_status_label = ctk.CTkLabel(
            self.canvas_toolbar,
            text="Draw mode: drag to create a new box.",
            anchor="w",
            text_color=("gray40", "gray70"),
        )
        self.tool_status_label.grid(row=0, column=5, padx=(0, 12), pady=12, sticky="ew")

        self.canvas_container = ctk.CTkFrame(self.right_panel)
        self.canvas_container.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="nsew")
        self.canvas_container.grid_rowconfigure(1, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)

        self.canvas_placeholder_label = ctk.CTkLabel(
            self.canvas_container,
            text="Image and annotation canvas",
            anchor="w",
        )
        self.canvas_placeholder_label.grid(row=0, column=0, padx=12, pady=(12, 8), sticky="ew")

        self.canvas = tk.Canvas(
            self.canvas_container,
            background="#1f1f1f",
            highlightthickness=0,
            relief="flat",
            cursor="crosshair",
        )
        self.canvas.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_left_button_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_left_button_release)
        self.canvas.bind("<Button-3>", self._show_canvas_context_menu)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_canvas_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_canvas_mouse_wheel)
        self.canvas.bind("<ButtonPress-2>", self._on_middle_button_press)
        self.canvas.bind("<B2-Motion>", self._on_middle_button_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_middle_button_release)
        self.tool_selector.set("▣ Draw")
        self._refresh_history_buttons()
        self._draw_canvas_message("Select an image from the list to display it.")

    def run(self) -> None:
        self.mainloop()


def main() -> None:
    app = YoloAnnotatorApp()
    app.run()


if __name__ == "__main__":
    main()
