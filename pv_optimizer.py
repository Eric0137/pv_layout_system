"""
PV Layout Optimizer
===================
Optimisation engine for maximising solar panel placement on detected roof area.
Uses grid-based placement with constraint satisfaction and greedy optimisation.
Accounts for spacing, tilt shading, walkways, and obstacle avoidance.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from config import PANEL_SPEC, LAYOUT, PERFORMANCE


class Panel:
    """Represents a single solar panel placement."""

    def __init__(
        self,
        x_m: float,
        y_m: float,
        width_m: float,
        height_m: float,
        orientation: str = "portrait",
        panel_id: int = 0,
        string_id: int = 0
    ):
        self.x_m = x_m               # Top-left x in meters
        self.y_m = y_m               # Top-left y in meters
        self.width_m = width_m       # Panel width on roof (meters)
        self.height_m = height_m     # Panel height on roof (meters)
        self.orientation = orientation
        self.panel_id = panel_id
        self.string_id = string_id

    @property
    def center_x(self) -> float:
        return self.x_m + self.width_m / 2

    @property
    def center_y(self) -> float:
        return self.y_m + self.height_m / 2

    @property
    def area_m2(self) -> float:
        return self.width_m * self.height_m

    def corners_m(self) -> List[Tuple[float, float]]:
        """Return corners as [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]."""
        return [
            (self.x_m, self.y_m),
            (self.x_m + self.width_m, self.y_m),
            (self.x_m + self.width_m, self.y_m + self.height_m),
            (self.x_m, self.y_m + self.height_m),
        ]

    def to_pixel_rect(
        self, scale_m_per_px: float, offset_x: int = 0, offset_y: int = 0
    ) -> Tuple[int, int, int, int]:
        """Convert to pixel coordinates (x, y, w, h)."""
        px = int(self.x_m / scale_m_per_px) + offset_x
        py = int(self.y_m / scale_m_per_px) + offset_y
        pw = int(self.width_m / scale_m_per_px)
        ph = int(self.height_m / scale_m_per_px)
        return (px, py, pw, ph)


class PVLayoutOptimizer:
    """
    Optimises PV panel placement on a given roof area.

    Strategy:
    1. Convert usable area mask to a placement grid
    2. Apply grid-based panel placement (portrait + landscape)
    3. Check each position against usable area mask
    4. Maximise panel count while respecting all constraints
    5. Group panels into electrical strings
    """

    def __init__(self):
        self.panels: List[Panel] = []
        self.panel_spec = PANEL_SPEC
        self.layout_params = LAYOUT
        self.total_capacity_kwp = 0.0
        self.roof_utilisation = 0.0
        self.strings: List[List[int]] = []

    def optimize_layout(
        self,
        usable_area_mask: np.ndarray,
        scale_m_per_px: float,
        orientation: str = "auto",
        tilt_deg: float = LAYOUT["tilt_angle_deg"],
        add_walkway: bool = True,
        walkway_interval: int = 10
    ) -> List[Panel]:
        """
        Main optimisation routine.

        Args:
            usable_area_mask: Binary mask (255=usable, 0=blocked)
            scale_m_per_px: Meters per pixel conversion factor
            orientation: "portrait", "landscape", or "auto" (tries both)
            tilt_deg: Panel tilt angle in degrees
            add_walkway: Whether to add maintenance walkways
            walkway_interval: Add walkway every N rows

        Returns:
            List of optimally placed Panel objects
        """
        if orientation == "auto":
            # Try both orientations and pick the one with more panels
            panels_portrait = self._place_panels(
                usable_area_mask, scale_m_per_px,
                "portrait", tilt_deg, add_walkway, walkway_interval
            )
            panels_landscape = self._place_panels(
                usable_area_mask, scale_m_per_px,
                "landscape", tilt_deg, add_walkway, walkway_interval
            )

            if len(panels_portrait) >= len(panels_landscape):
                self.panels = panels_portrait
                best_orientation = "portrait"
            else:
                self.panels = panels_landscape
                best_orientation = "landscape"

            print(
                f"[OPTIMIZER] Auto-selected {best_orientation}: "
                f"{len(self.panels)} panels "
                f"(portrait={len(panels_portrait)}, "
                f"landscape={len(panels_landscape)})"
            )
        else:
            self.panels = self._place_panels(
                usable_area_mask, scale_m_per_px,
                orientation, tilt_deg, add_walkway, walkway_interval
            )

        # Calculate metrics
        self._calculate_metrics(usable_area_mask, scale_m_per_px)

        # Assign string IDs
        self._assign_strings()

        return self.panels

    def _place_panels(
        self,
        mask: np.ndarray,
        scale: float,
        orientation: str,
        tilt_deg: float,
        add_walkway: bool,
        walkway_interval: int
    ) -> List[Panel]:
        """Core panel placement algorithm."""
        # Panel dimensions based on orientation
        if orientation == "portrait":
            panel_w_m = PANEL_SPEC["width_m"]    # shorter side horizontal
            panel_h_m = PANEL_SPEC["length_m"]   # longer side vertical
        else:
            panel_w_m = PANEL_SPEC["length_m"]   # longer side horizontal
            panel_h_m = PANEL_SPEC["width_m"]    # shorter side vertical

        # Account for tilt angle (projected footprint on roof)
        tilt_rad = np.radians(tilt_deg)
        effective_h_m = panel_h_m * np.cos(tilt_rad)

        # Shading distance (row spacing to avoid inter-row shading)
        # For Malaysia (~3-5° latitude), shading distance is relatively small
        shade_distance_m = panel_h_m * np.sin(tilt_rad) * np.tan(
            np.radians(90 - 3 - tilt_deg)  # ~3° latitude for Malaysia
        )
        # Minimum row gap is the larger of shade distance or specified gap
        row_gap_m = max(
            LAYOUT["inter_row_gap_m"],
            shade_distance_m
        )

        col_gap_m = LAYOUT["inter_col_gap_m"]

        # Step sizes in meters
        step_x_m = panel_w_m + col_gap_m
        step_y_m = effective_h_m + row_gap_m

        # Walkway step (in number of rows)
        walkway_h_m = LAYOUT["walkway_width_m"]

        # Get mask bounding box
        h_px, w_px = mask.shape
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return []

        min_y_px, max_y_px = coords[0].min(), coords[0].max()
        min_x_px, max_x_px = coords[1].min(), coords[1].max()

        # Convert to meters
        start_x_m = min_x_px * scale
        start_y_m = min_y_px * scale
        end_x_m = max_x_px * scale
        end_y_m = max_y_px * scale

        panels = []
        panel_id = 0
        row_count = 0

        y_m = start_y_m
        while y_m + effective_h_m <= end_y_m:
            # Add walkway every N rows
            if add_walkway and row_count > 0 and row_count % walkway_interval == 0:
                y_m += walkway_h_m

            x_m = start_x_m
            while x_m + panel_w_m <= end_x_m:
                # Check if panel fits entirely within usable area
                if self._panel_fits_in_mask(
                    x_m, y_m, panel_w_m, effective_h_m, mask, scale
                ):
                    panel = Panel(
                        x_m=x_m,
                        y_m=y_m,
                        width_m=panel_w_m,
                        height_m=effective_h_m,
                        orientation=orientation,
                        panel_id=panel_id
                    )
                    panels.append(panel)
                    panel_id += 1

                x_m += step_x_m

            y_m += step_y_m
            row_count += 1

        return panels

    def _panel_fits_in_mask(
        self,
        x_m: float, y_m: float,
        w_m: float, h_m: float,
        mask: np.ndarray,
        scale: float
    ) -> bool:
        """Check if a panel rectangle fits entirely within the usable area mask."""
        # Convert to pixel coordinates
        x1 = int(x_m / scale)
        y1 = int(y_m / scale)
        x2 = int((x_m + w_m) / scale)
        y2 = int((y_m + h_m) / scale)

        # Clamp to image bounds
        h_px, w_px = mask.shape
        x1 = max(0, min(x1, w_px - 1))
        y1 = max(0, min(y1, h_px - 1))
        x2 = max(0, min(x2, w_px))
        y2 = max(0, min(y2, h_px))

        if x2 <= x1 or y2 <= y1:
            return False

        # Check all pixels in panel area are usable
        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        # Require >95% of panel area to be usable (tolerance for rounding)
        return np.mean(roi > 0) > 0.95

    def _calculate_metrics(self, mask: np.ndarray, scale: float):
        """Calculate system capacity and roof utilisation."""
        n_panels = len(self.panels)
        self.total_capacity_kwp = (
            n_panels * PANEL_SPEC["rated_power_wp"] / 1000.0
        )

        # Roof utilisation
        total_panel_area = sum(p.area_m2 for p in self.panels)
        usable_area_px = np.count_nonzero(mask)
        usable_area_m2 = usable_area_px * (scale ** 2)

        self.roof_utilisation = (
            total_panel_area / usable_area_m2 * 100
            if usable_area_m2 > 0 else 0
        )

    def _assign_strings(self):
        """Group panels into electrical strings for inverter sizing."""
        if not self.panels:
            return

        max_per_string = LAYOUT["max_string_length"]
        min_per_string = LAYOUT["min_string_length"]

        # Sort panels by row (y) then column (x)
        sorted_panels = sorted(
            self.panels, key=lambda p: (round(p.y_m, 1), p.x_m)
        )

        self.strings = []
        current_string = []
        string_id = 0

        for panel in sorted_panels:
            current_string.append(panel.panel_id)
            panel.string_id = string_id

            if len(current_string) >= max_per_string:
                self.strings.append(current_string)
                current_string = []
                string_id += 1

        # Handle remaining panels
        if current_string:
            if len(current_string) >= min_per_string:
                self.strings.append(current_string)
            elif self.strings:
                # Redistribute to last string if too few
                self.strings[-1].extend(current_string)
                for pid in current_string:
                    for p in self.panels:
                        if p.panel_id == pid:
                            p.string_id = len(self.strings) - 1
            else:
                self.strings.append(current_string)

    def get_layout_summary(self) -> Dict:
        """Return comprehensive layout summary."""
        return {
            "total_panels": len(self.panels),
            "total_capacity_kwp": round(self.total_capacity_kwp, 2),
            "total_capacity_kw": round(self.total_capacity_kwp, 2),
            "panel_rated_power_wp": PANEL_SPEC["rated_power_wp"],
            "panel_dimensions_m": f"{PANEL_SPEC['length_m']} x {PANEL_SPEC['width_m']}",
            "orientation": (
                self.panels[0].orientation if self.panels else "N/A"
            ),
            "roof_utilisation_pct": round(self.roof_utilisation, 1),
            "number_of_strings": len(self.strings),
            "panels_per_string": [len(s) for s in self.strings],
            "tilt_angle_deg": LAYOUT["tilt_angle_deg"],
            "row_gap_m": LAYOUT["inter_row_gap_m"],
            "edge_setback_m": LAYOUT["min_edge_setback_m"],
        }

    def get_inverter_recommendation(self) -> Dict:
        """
        Recommend inverter sizing based on total capacity.
        Uses standard DC/AC ratio of 1.1-1.3 for Malaysia.
        """
        dc_capacity_kw = self.total_capacity_kwp
        # Target DC/AC ratio of 1.2
        ac_capacity_kw = dc_capacity_kw / 1.2

        # Common inverter sizes (kW)
        inverter_sizes = [3, 5, 6, 8, 10, 12, 15, 20, 25, 30, 33, 36, 50, 60]

        # Find optimal inverter combination
        best_combo = None
        best_waste = float('inf')

        for size in inverter_sizes:
            n = int(np.ceil(ac_capacity_kw / size))
            total = n * size
            waste = total - ac_capacity_kw
            dc_ac_ratio = dc_capacity_kw / total

            if 1.0 <= dc_ac_ratio <= 1.35 and waste < best_waste:
                best_combo = {"size_kw": size, "quantity": n, "total_ac_kw": total}
                best_waste = waste

        if best_combo is None:
            # Fallback: single inverter matching capacity
            best_combo = {
                "size_kw": round(ac_capacity_kw, 1),
                "quantity": 1,
                "total_ac_kw": round(ac_capacity_kw, 1)
            }

        dc_ac_ratio = dc_capacity_kw / best_combo["total_ac_kw"]

        return {
            "dc_capacity_kwp": round(dc_capacity_kw, 2),
            "recommended_ac_kw": best_combo["total_ac_kw"],
            "inverter_size_kw": best_combo["size_kw"],
            "inverter_quantity": best_combo["quantity"],
            "dc_ac_ratio": round(dc_ac_ratio, 2),
            "strings_count": len(self.strings),
        }

    def get_panel_positions_px(
        self,
        scale_m_per_px: float,
        offset_x: int = 0,
        offset_y: int = 0
    ) -> List[Tuple[int, int, int, int]]:
        """Get all panel positions as pixel rectangles for visualisation."""
        return [
            p.to_pixel_rect(scale_m_per_px, offset_x, offset_y)
            for p in self.panels
        ]
