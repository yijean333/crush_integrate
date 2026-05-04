"""
LTE-TrackGuard Annotation Tool
================================
Annotate collision events in TU-DAT with:
  - Layer 1: Event window (corrected start/end frame)
  - Layer 2: Bounding boxes for involved vehicles

Saves back to TU-DAT JSON format with impact_region added.
"""

import copy
import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("Install Pillow first:  pip install Pillow")

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
TUDAT_DIR  = BASE_DIR / "TU-DAT"
FRAMES_DIR = BASE_DIR / "tu-dat-frames"

# ── Constants ─────────────────────────────────────────────────────────
CANVAS_W   = 900
CANVAS_H   = 540
BOX_COLORS = ["#FF4444", "#4488FF", "#44CC44", "#FF8800", "#CC44CC", "#00BBBB",
              "#FFFF00", "#FF44AA", "#88FF44", "#44FFFF"]


# ── Data ──────────────────────────────────────────────────────────────
class BBox:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, color: str):
        self.x1, self.y1 = min(x1, x2), min(y1, y2)
        self.x2, self.y2 = max(x1, x2), max(y1, y2)
        self.color = color

    def to_normalized(self, img_w: int, img_h: int,
                      scale: float, off_x: float, off_y: float) -> dict:
        """Convert canvas-pixel coords → normalized (cx, cy, w, h)."""
        ix1 = max(0.0, (self.x1 - off_x) / scale)
        iy1 = max(0.0, (self.y1 - off_y) / scale)
        ix2 = min(img_w, (self.x2 - off_x) / scale)
        iy2 = min(img_h, (self.y2 - off_y) / scale)
        cx = (ix1 + ix2) / 2 / img_w
        cy = (iy1 + iy2) / 2 / img_h
        w  = (ix2 - ix1) / img_w
        h  = (iy2 - iy1) / img_h
        return {"cx": round(cx, 4), "cy": round(cy, 4),
                "w":  round(w,  4), "h":  round(h,  4)}


# ── Main App ──────────────────────────────────────────────────────────
class AnnotatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LTE-TrackGuard Annotation Tool")
        self.root.geometry("1300x800")
        self.root.minsize(900, 600)

        # ── Runtime state ─────────────────────────────────────────────
        self.annotations: dict = {}          # video_key → full JSON dict
        self._event_index: List[Tuple[str, int]] = []  # parallel to listbox

        self.current_video: Optional[str] = None
        self.current_event_id: Optional[int] = None
        self.frame_files: List[Path] = []
        self.total_frames: int = 0
        self.current_frame_idx: int = 1

        self.img_w: int = 1
        self.img_h: int = 1
        self.scale: float = 1.0
        self.off_x: float = 0.0
        self.off_y: float = 0.0
        self._photo: Optional[ImageTk.PhotoImage] = None

        # bbox drawing
        self.bboxes: List[BBox] = []
        self._drawing: bool = False
        self._draw_start: Optional[Tuple[float, float]] = None
        self._live_rect: Optional[int] = None
        self._next_color_idx: int = 0

        # custom (non-TU-DAT) frames: {list_index: Path(frames_subfolder)}
        self._custom_frames: dict = {}

        self._build_menu()
        self._build_ui()
        self._load_all_annotations()

    # ── Menu bar ──────────────────────────────────────────────────────
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Frames Folder…", command=self._open_frames_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    # ── UI construction ───────────────────────────────────────────────
    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── Sidebar ───────────────────────────────────────────────────
        sidebar = ttk.Frame(self.root, width=230, padding=(5, 5))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.rowconfigure(1, weight=1)

        ttk.Label(sidebar, text="Videos / Events",
                  font=("", 10, "bold")).grid(row=0, column=0, sticky="w")

        lf = ttk.Frame(sidebar)
        lf.grid(row=1, column=0, sticky="nsew", pady=4)
        sidebar.rowconfigure(1, weight=1)
        sidebar.columnconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)

        sb = ttk.Scrollbar(lf, orient="vertical")
        self.event_list = tk.Listbox(lf, yscrollcommand=sb.set,
                                     selectmode="single", activestyle="dotbox",
                                     font=("Consolas", 9))
        sb.config(command=self.event_list.yview)
        self.event_list.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.event_list.bind("<<ListboxSelect>>", self._on_select)

        ttk.Separator(sidebar, orient="horizontal").grid(
            row=2, column=0, sticky="ew", pady=6)

        self.info_var = tk.StringVar(value="—")
        ttk.Label(sidebar, textvariable=self.info_var,
                  wraplength=215, justify="left",
                  font=("", 9)).grid(row=3, column=0, sticky="w")

        # ── Right pane ────────────────────────────────────────────────
        right = ttk.Frame(self.root, padding=(5, 5))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        # Canvas
        cf = ttk.LabelFrame(right, text="Frame")
        cf.grid(row=0, column=0, sticky="nsew")
        cf.columnconfigure(0, weight=1)
        cf.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(cf, bg="#1e1e1e", cursor="crosshair",
                                width=CANVAS_W, height=CANVAS_H,
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>",   self._mouse_down)
        self.canvas.bind("<B1-Motion>",        self._mouse_drag)
        self.canvas.bind("<ButtonRelease-1>",  self._mouse_up)
        self.canvas.bind("<Configure>",        self._on_canvas_resize)

        # ── Controls area ─────────────────────────────────────────────
        ctrl = ttk.Frame(right)
        ctrl.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ctrl.columnconfigure(0, weight=1)

        # Row 1 — Frame navigation
        nav = ttk.LabelFrame(ctrl, text="Frame Navigation")
        nav.grid(row=0, column=0, sticky="ew", pady=2)
        nav.columnconfigure(2, weight=1)

        ttk.Label(nav, text="Frame:").grid(row=0, column=0, padx=(6, 2))
        self.frame_var = tk.StringVar(value="1")
        fe = ttk.Entry(nav, textvariable=self.frame_var, width=7)
        fe.grid(row=0, column=1, padx=2)
        fe.bind("<Return>", lambda e: self._jump_to_frame())

        ttk.Button(nav, text="◀", width=3,
                   command=lambda: self._step(-1)).grid(row=0, column=2, padx=2, sticky="e")

        self.slider = ttk.Scale(nav, from_=1, to=1000, orient="horizontal",
                                command=self._on_slider_move)
        self.slider.grid(row=0, column=3, sticky="ew", padx=4)
        nav.columnconfigure(3, weight=1)

        ttk.Button(nav, text="▶", width=3,
                   command=lambda: self._step(1)).grid(row=0, column=4, padx=2)

        self.total_var = tk.StringVar(value="/ —")
        ttk.Label(nav, textvariable=self.total_var).grid(row=0, column=5, padx=(2, 6))

        # Row 2 — Event window
        ew = ttk.LabelFrame(ctrl, text="Event Window  (will overwrite GT start_frame / end_frame)")
        ew.grid(row=1, column=0, sticky="ew", pady=2)

        ttk.Label(ew, text="Start frame:").grid(row=0, column=0, padx=(6, 2))
        self.win_start = tk.StringVar()
        ttk.Entry(ew, textvariable=self.win_start, width=8).grid(row=0, column=1, padx=2)
        ttk.Button(ew, text="← current",
                   command=lambda: self.win_start.set(
                       str(self.current_frame_idx))).grid(row=0, column=2, padx=6)

        ttk.Separator(ew, orient="vertical").grid(row=0, column=3, sticky="ns", padx=6)

        ttk.Label(ew, text="End frame:").grid(row=0, column=4, padx=(0, 2))
        self.win_end = tk.StringVar()
        ttk.Entry(ew, textvariable=self.win_end, width=8).grid(row=0, column=5, padx=2)
        ttk.Button(ew, text="← current",
                   command=lambda: self.win_end.set(
                       str(self.current_frame_idx))).grid(row=0, column=6, padx=6)

        # Row 3 — Bounding boxes
        bb_row = ttk.LabelFrame(ctrl, text="Bounding Boxes  (click-drag on canvas to draw)")
        bb_row.grid(row=2, column=0, sticky="ew", pady=2)

        self.box_count_var = tk.StringVar(value="Boxes: 0")
        ttk.Label(bb_row, textvariable=self.box_count_var,
                  font=("", 9, "bold")).pack(side="left", padx=8)

        ttk.Button(bb_row, text="Undo last",
                   command=self._undo).pack(side="left", padx=4)
        ttk.Button(bb_row, text="Clear all",
                   command=self._clear_all).pack(side="left", padx=4)

        self.legend_frame = ttk.Frame(bb_row)
        self.legend_frame.pack(side="left", padx=10)

        # Row 4 — Save + status
        save_row = ttk.Frame(ctrl)
        save_row.grid(row=3, column=0, sticky="ew", pady=(4, 0))

        ttk.Button(save_row, text="  Save annotation as…  ",
                   command=self._save).pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Select a video / event from the list.")
        ttk.Label(save_row, textvariable=self.status_var,
                  foreground="#888888").pack(side="left", padx=8)

    # ── Annotation loading ────────────────────────────────────────────
    def _load_all_annotations(self):
        self.annotations.clear()
        self._event_index.clear()
        self.event_list.delete(0, tk.END)

        for txt_file in sorted(TUDAT_DIR.glob("*.txt")):
            if "skip" in txt_file.name.lower():
                continue
            m = re.match(r'(v\d+)', txt_file.name)
            if not m:
                continue
            vkey = m.group(1)

            try:
                data = json.loads(txt_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            if data.get("ignore"):
                continue

            self.annotations[vkey] = data

            for ev in data.get("events", []):
                if ev.get("event_type") != "collision":
                    continue
                has = "✓" if ev.get("impact_region") else " "
                label = (f"{has} {vkey:>4} | ev{ev['event_id']} "
                         f"| f{ev['start_frame']}–{ev['end_frame']}")
                self.event_list.insert(tk.END, label)
                self._event_index.append((vkey, ev["event_id"]))

    # ── Open arbitrary frames folder ──────────────────────────────────
    def _open_frames_folder(self):
        root_dir = filedialog.askdirectory(
            title="Select root folder (containing subfolders with frames)")
        if not root_dir:
            return
        root_path = Path(root_dir)

        # Scan for subfolders that contain frame images
        subfolders = sorted([
            d for d in root_path.iterdir()
            if d.is_dir() and (
                any(d.glob("frame_*.jpg")) or
                any(d.glob("frame_*.png")) or
                any(d.glob("*.jpg")) or
                any(d.glob("*.png"))
            )
        ])

        if not subfolders:
            messagebox.showwarning("No frames found",
                f"No subfolders with images found in:\n{root_path}")
            return

        # Add separator in listbox
        sep_idx = self.event_list.size()
        self.event_list.insert(tk.END, f"── {root_path.name} ──")
        self._event_index.append(None)  # separator, not selectable

        for folder in subfolders:
            idx = self.event_list.size()
            self.event_list.insert(tk.END, f"  📁 {folder.name}")
            self._event_index.append(('__custom__', str(folder)))
            self._custom_frames[idx] = folder

        self.status_var.set(f"Loaded {len(subfolders)} folders from {root_path.name}")

    # ── Event selection ───────────────────────────────────────────────
    def _on_select(self, _event=None):
        sel = self.event_list.curselection()
        if not sel:
            return
        entry = self._event_index[sel[0]]
        if entry is None:
            return  # separator
        if entry[0] == '__custom__':
            self._load_custom_frames(Path(entry[1]))
        else:
            vkey, ev_id = entry
            self._load_event(vkey, ev_id)

    def _load_custom_frames(self, folder: Path):
        """Load frames from an arbitrary folder (no TU-DAT JSON needed)."""
        ffiles = sorted(folder.glob("frame_*.jpg"))
        if not ffiles:
            ffiles = sorted(folder.glob("frame_*.png"))
        if not ffiles:
            ffiles = sorted(folder.glob("*.jpg"))
        if not ffiles:
            ffiles = sorted(folder.glob("*.png"))
        if not ffiles:
            self.status_var.set(f"No image files found in {folder}")
            return

        self.current_video    = folder.name
        self.current_event_id = None
        self.bboxes.clear()
        self._next_color_idx  = 0
        self._pending_restore = []
        self._update_legend()

        self.frame_files  = ffiles
        self.total_frames = len(ffiles)
        self.total_var.set(f"/ {self.total_frames}")
        self.slider.config(to=self.total_frames)

        # Reset window fields
        self.win_start.set("1")
        self.win_end.set(str(self.total_frames))

        self.info_var.set(f"{folder.name}\n{self.total_frames} frames\n(custom folder)")
        self._go_to(1)

    def _load_event(self, vkey: str, ev_id: int):
        self.current_video    = vkey
        self.current_event_id = ev_id
        self.bboxes.clear()
        self._next_color_idx = 0
        self._update_legend()

        data = self.annotations.get(vkey, {})
        ev   = next((e for e in data.get("events", [])
                     if e["event_id"] == ev_id), None)
        if ev is None:
            return

        # Sidebar info
        info = (f"{vkey}  |  Event {ev_id}\n"
                f"Severity: {ev.get('severity', '?')}\n"
                f"GT: f{ev['start_frame']} – {ev['end_frame']}")
        ir = ev.get("impact_region")
        if ir:
            info += f"\n[Annotated ✓ @ f{ir.get('annotated_frame','?')}]"
        self.info_var.set(info)

        # Pre-fill window fields
        self.win_start.set(str(ir["window_start_frame"] if ir and "window_start_frame" in ir
                               else ev["start_frame"]))
        self.win_end.set(str(ir["window_end_frame"] if ir and "window_end_frame" in ir
                             else ev["end_frame"]))

        # Load frame list
        frames_dir = FRAMES_DIR / vkey
        if not frames_dir.exists():
            self.status_var.set(f"⚠  Frame dir not found: {frames_dir}")
            return

        ffiles = sorted(frames_dir.glob("frame_*.jpg"))
        if not ffiles:
            ffiles = sorted(frames_dir.glob("frame_*.png"))
        if not ffiles:
            self.status_var.set(f"⚠  No frames in {frames_dir}")
            return

        self.frame_files  = ffiles
        self.total_frames = len(ffiles)
        self.total_var.set(f"/ {self.total_frames}")
        self.slider.config(to=self.total_frames)

        # Restore existing boxes or jump to GT start_frame
        start_frame = ev["start_frame"]
        if ir and ir.get("annotated_frame") and ir.get("vehicles"):
            self._pending_restore = ir["vehicles"]
            self._go_to(ir["annotated_frame"])
        else:
            self._pending_restore = []
            self._go_to(min(start_frame, self.total_frames))

    # ── Frame navigation ──────────────────────────────────────────────
    def _go_to(self, idx: int):
        idx = max(1, min(idx, self.total_frames))
        self.current_frame_idx = idx
        self.frame_var.set(str(idx))
        self.slider.set(idx)
        self._show_frame(idx)

    def _show_frame(self, idx: int):
        if not self.frame_files:
            return

        path = self.frame_files[idx - 1]
        try:
            img = Image.open(path)
        except Exception as e:
            self.status_var.set(f"⚠  {e}")
            return

        self.img_w, self.img_h = img.size

        cw = max(self.canvas.winfo_width(),  CANVAS_W)
        ch = max(self.canvas.winfo_height(), CANVAS_H)
        self.scale = min(cw / self.img_w, ch / self.img_h)
        nw = int(self.img_w * self.scale)
        nh = int(self.img_h * self.scale)
        self.off_x = (cw - nw) / 2
        self.off_y = (ch - nh) / 2

        resized   = img.resize((nw, nh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(self.off_x, self.off_y,
                                 anchor="nw", image=self._photo)

        # Restore pending boxes from saved annotation
        if hasattr(self, "_pending_restore") and self._pending_restore:
            for v in self._pending_restore:
                color = BOX_COLORS[self._next_color_idx % len(BOX_COLORS)]
                self._next_color_idx += 1
                # normalized → canvas pixels
                cx_px = v["cx"] * self.img_w * self.scale + self.off_x
                cy_px = v["cy"] * self.img_h * self.scale + self.off_y
                hw = v["w"] * self.img_w * self.scale / 2
                hh = v["h"] * self.img_h * self.scale / 2
                self.bboxes.append(BBox(cx_px - hw, cy_px - hh,
                                        cx_px + hw, cy_px + hh, color))
            self._pending_restore = []

        self._redraw_boxes()
        self.status_var.set(
            f"{self.current_video}  |  Frame {idx} / {self.total_frames}")

    def _on_slider_move(self, val):
        idx = int(float(val))
        if idx != self.current_frame_idx:
            self.current_frame_idx = idx
            self.frame_var.set(str(idx))
            self._show_frame(idx)

    def _jump_to_frame(self):
        try:
            self._go_to(int(self.frame_var.get()))
        except ValueError:
            pass

    def _step(self, delta: int):
        self._go_to(self.current_frame_idx + delta)

    def _on_canvas_resize(self, _event=None):
        if self.frame_files and self.current_frame_idx:
            self._show_frame(self.current_frame_idx)

    # ── Bounding box drawing ──────────────────────────────────────────
    def _mouse_down(self, event):
        self._drawing    = True
        self._draw_start = (event.x, event.y)
        color = BOX_COLORS[self._next_color_idx % len(BOX_COLORS)]
        self._current_color = color
        self._live_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=color, width=2, dash=(4, 4), tags="live")

    def _mouse_drag(self, event):
        if self._drawing and self._live_rect and self._draw_start:
            self.canvas.coords(self._live_rect,
                               self._draw_start[0], self._draw_start[1],
                               event.x, event.y)

    def _mouse_up(self, event):
        if not self._drawing or not self._draw_start:
            return
        self._drawing = False
        self.canvas.delete("live")

        x0, y0 = self._draw_start
        x1, y1 = event.x, event.y
        if abs(x1 - x0) < 6 or abs(y1 - y0) < 6:
            return   # too small – ignore accidental click

        self.bboxes.append(BBox(x0, y0, x1, y1, self._current_color))
        self._next_color_idx += 1
        self._redraw_boxes()

    def _redraw_boxes(self):
        self.canvas.delete("bbox")
        for i, bb in enumerate(self.bboxes):
            self.canvas.create_rectangle(bb.x1, bb.y1, bb.x2, bb.y2,
                                         outline=bb.color, width=2, tags="bbox")
            # Label at top-left of box
            self.canvas.create_rectangle(bb.x1, bb.y1, bb.x1 + 28, bb.y1 + 18,
                                         fill=bb.color, outline="", tags="bbox")
            self.canvas.create_text(bb.x1 + 14, bb.y1 + 9,
                                    text=f"V{i+1}", fill="white",
                                    font=("", 9, "bold"), tags="bbox")
        self.box_count_var.set(f"Boxes: {len(self.bboxes)}")
        self._update_legend()

    def _update_legend(self):
        for w in self.legend_frame.winfo_children():
            w.destroy()
        for i, bb in enumerate(self.bboxes):
            f = ttk.Frame(self.legend_frame)
            f.pack(side="left", padx=3)
            tk.Label(f, text="■", fg=bb.color, font=("", 13)).pack(side="left")
            ttk.Label(f, text=f"V{i+1}").pack(side="left")

    def _undo(self):
        if self.bboxes:
            self.bboxes.pop()
            self._next_color_idx = max(0, self._next_color_idx - 1)
            self._redraw_boxes()

    def _clear_all(self):
        self.bboxes.clear()
        self._next_color_idx = 0
        self._redraw_boxes()

    # ── Save ─────────────────────────────────────────────────────────
    def _save(self):
        if not self.current_video:
            messagebox.showwarning("No event", "Select a video / event first.")
            return
        # Custom folder mode: save as new TU-DAT JSON
        if self.current_event_id is None:
            self._save_custom()
            return
        if not self.bboxes:
            messagebox.showwarning("No boxes", "Draw at least one bounding box first.")
            return

        try:
            ws = int(self.win_start.get())
            we = int(self.win_end.get())
        except ValueError:
            messagebox.showerror("Invalid window", "Enter valid integer frame numbers.")
            return
        if ws >= we:
            messagebox.showerror("Invalid window", "Start frame must be < end frame.")
            return

        vehicles = [bb.to_normalized(self.img_w, self.img_h,
                                     self.scale, self.off_x, self.off_y)
                    for bb in self.bboxes]

        data = copy.deepcopy(self.annotations[self.current_video])
        for ev in data["events"]:
            if ev["event_id"] == self.current_event_id:
                ev["start_frame"]  = ws   # overwrite
                ev["end_frame"]    = we   # overwrite
                ev["impact_region"] = {
                    "window_start_frame": ws,
                    "window_end_frame":   we,
                    "annotated_frame":    self.current_frame_idx,
                    "vehicles":           vehicles,
                }
                break

        # Save dialog
        save_path = filedialog.asksaveasfilename(
            title="Save annotation as",
            initialdir=str(TUDAT_DIR),
            initialfile=f"{self.current_video}.txt",
            defaultextension=".txt",
            filetypes=[("JSON text", "*.txt"), ("All", "*.*")],
        )
        if not save_path:
            return

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.status_var.set(f"Saved → {Path(save_path).name}")
        messagebox.showinfo("Saved", f"Annotation saved:\n{save_path}")

        # Refresh sidebar list
        self._load_all_annotations()

    def _save_custom(self):
        """Save annotation for a custom (non-TU-DAT) folder as a new JSON."""
        if not self.bboxes:
            messagebox.showwarning("No boxes", "Draw at least one bounding box first.")
            return
        try:
            ws = int(self.win_start.get())
            we = int(self.win_end.get())
        except ValueError:
            messagebox.showerror("Invalid window", "Enter valid integer frame numbers.")
            return
        if ws >= we:
            messagebox.showerror("Invalid window", "Start frame must be < end frame.")
            return

        vehicles = [bb.to_normalized(self.img_w, self.img_h,
                                     self.scale, self.off_x, self.off_y)
                    for bb in self.bboxes]

        data = {
            "video_name": f"{self.current_video}.mp4",
            "dataset": "TU-DAT",
            "num_frames": self.total_frames,
            "fps": 30,
            "ignore": False,
            "metadata": {"ego_involve": True, "night": False},
            "events": [{
                "event_id": 1,
                "event_type": "collision",
                "start_frame": ws,
                "end_frame": we,
                "primary_object_type": "vehicle",
                "severity": "moderate",
                "impact_region": {
                    "window_start_frame": ws,
                    "window_end_frame": we,
                    "annotated_frame": self.current_frame_idx,
                    "vehicles": vehicles,
                }
            }]
        }

        save_path = filedialog.asksaveasfilename(
            title="Save annotation as",
            initialdir=str(TUDAT_DIR),
            initialfile=f"{self.current_video}.txt",
            defaultextension=".txt",
            filetypes=[("JSON text", "*.txt"), ("All", "*.*")],
        )
        if not save_path:
            return

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.status_var.set(f"Saved → {Path(save_path).name}")
        messagebox.showinfo("Saved", f"Annotation saved:\n{save_path}")
        self._load_all_annotations()


# ── Entry point ───────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    app  = AnnotatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
