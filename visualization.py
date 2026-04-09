"""
Visualization Module
====================
Renders PV layout overlays, energy charts, and system diagrams.
Uses OpenCV for image overlays and matplotlib for charts.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple
from config import PANEL_SPEC


class LayoutVisualizer:
    """Renders PV panel layout overlays on roof images."""

    # Color palette (BGR for OpenCV)
    COLORS = {
        "panel": (0, 180, 0),          # Green panels
        "panel_fill": (0, 140, 0),     # Darker green fill
        "roof_boundary": (255, 0, 0),   # Blue roof outline
        "obstacle": (0, 0, 255),        # Red obstacles
        "usable_area": (200, 200, 0),   # Cyan usable area
        "text": (255, 255, 255),        # White text
        "bg": (40, 40, 40),             # Dark background
        "walkway": (0, 165, 255),       # Orange walkway
    }

    STRING_COLORS = [
        (0, 180, 0), (180, 0, 0), (0, 0, 180),
        (180, 180, 0), (180, 0, 180), (0, 180, 180),
        (100, 200, 100), (200, 100, 100), (100, 100, 200),
    ]

    @staticmethod
    def draw_layout_overlay(
        image: np.ndarray,
        panels: list,
        scale_m_per_px: float,
        roof_contour: Optional[np.ndarray] = None,
        obstacles: Optional[List[Dict]] = None,
        usable_mask: Optional[np.ndarray] = None,
        show_strings: bool = True,
        show_labels: bool = True,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Draw complete PV layout overlay on roof image.

        Args:
            image: Original roof image (BGR)
            panels: List of Panel objects from optimizer
            scale_m_per_px: Scale factor
            roof_contour: Roof boundary contour
            obstacles: Detected obstacle dicts
            usable_mask: Binary usable area mask
            show_strings: Color panels by string ID
            show_labels: Show panel numbers
            alpha: Overlay transparency

        Returns:
            Annotated image with layout overlay
        """
        overlay = image.copy()
        output = image.copy()
        h, w = image.shape[:2]

        # Draw usable area (semi-transparent)
        if usable_mask is not None:
            usable_overlay = np.zeros_like(overlay)
            usable_overlay[usable_mask > 0] = LayoutVisualizer.COLORS["usable_area"]
            cv2.addWeighted(usable_overlay, 0.15, overlay, 1.0, 0, overlay)

        # Draw roof boundary
        if roof_contour is not None:
            cv2.drawContours(
                overlay, [roof_contour], -1,
                LayoutVisualizer.COLORS["roof_boundary"], 3
            )

        # Draw obstacles
        if obstacles:
            for obs in obstacles:
                if obs.get("label", "") != "none":
                    contour = obs.get("contour")
                    if contour is not None:
                        cv2.drawContours(
                            overlay, [contour], -1,
                            LayoutVisualizer.COLORS["obstacle"], 2
                        )
                        # Label
                        x, y, bw, bh = obs["bbox"]
                        label_text = f"{obs.get('label', '?')} ({obs.get('confidence', 0):.0%})"
                        cv2.putText(
                            overlay, label_text,
                            (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, LayoutVisualizer.COLORS["obstacle"], 1
                        )

        # Draw panels
        for panel in panels:
            px, py, pw, ph = panel.to_pixel_rect(scale_m_per_px)

            # Panel color by string
            if show_strings:
                color_idx = panel.string_id % len(LayoutVisualizer.STRING_COLORS)
                color = LayoutVisualizer.STRING_COLORS[color_idx]
            else:
                color = LayoutVisualizer.COLORS["panel"]

            # Filled rectangle
            cv2.rectangle(overlay, (px, py), (px + pw, py + ph), color, -1)
            # Border
            cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (255, 255, 255), 1)

            # Panel label
            if show_labels and pw > 20 and ph > 15:
                label = str(panel.panel_id + 1)
                font_scale = min(pw, ph) / 80
                font_scale = max(0.25, min(font_scale, 0.5))
                cv2.putText(
                    overlay, label,
                    (px + 3, py + ph - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255), 1
                )

        # Blend overlay
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Redraw borders on top (not blended)
        for panel in panels:
            px, py, pw, ph = panel.to_pixel_rect(scale_m_per_px)
            cv2.rectangle(output, (px, py), (px + pw, py + ph), (255, 255, 255), 1)

        return output

    @staticmethod
    def draw_info_panel(
        image: np.ndarray,
        layout_summary: Dict,
        energy_data: Dict,
        position: str = "right"
    ) -> np.ndarray:
        """Add an information panel beside the layout image."""
        h, w = image.shape[:2]
        panel_w = 350
        info = np.zeros((h, panel_w, 3), dtype=np.uint8)
        info[:] = LayoutVisualizer.COLORS["bg"]

        y_offset = 30
        line_height = 25

        def put_text(text, y, color=(255, 255, 255), scale=0.5, thickness=1):
            cv2.putText(info, text, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

        # Title
        put_text("PV LAYOUT SUMMARY", y_offset, (0, 200, 255), 0.7, 2)
        y_offset += 40

        # Layout info
        put_text(f"Total Panels: {layout_summary['total_panels']}", y_offset)
        y_offset += line_height
        put_text(f"Capacity: {layout_summary['total_capacity_kwp']} kWp", y_offset)
        y_offset += line_height
        put_text(f"Panel: {PANEL_SPEC['rated_power_wp']}W", y_offset)
        y_offset += line_height
        put_text(f"Orientation: {layout_summary['orientation']}", y_offset)
        y_offset += line_height
        put_text(f"Roof Usage: {layout_summary['roof_utilisation_pct']}%", y_offset)
        y_offset += line_height
        put_text(f"Strings: {layout_summary['number_of_strings']}", y_offset)
        y_offset += line_height
        put_text(f"Tilt: {layout_summary['tilt_angle_deg']} deg", y_offset)
        y_offset += 40

        # Energy info
        if energy_data:
            put_text("ENERGY ESTIMATE", y_offset, (0, 200, 255), 0.7, 2)
            y_offset += 35
            put_text(f"Region: {energy_data.get('region', 'N/A')}", y_offset)
            y_offset += line_height
            annual = energy_data.get('net_annual_yield_kwh', 0)
            put_text(f"Annual Yield: {annual:,.0f} kWh", y_offset)
            y_offset += line_height
            specific = energy_data.get('specific_yield_kwh_kwp', 0)
            put_text(f"Specific Yield: {specific:,.0f} kWh/kWp", y_offset)
            y_offset += line_height
            pr = energy_data.get('performance_ratio', 0)
            put_text(f"Perf. Ratio: {pr:.1%}", y_offset)
            y_offset += line_height
            daily = energy_data.get('daily_avg_yield_kwh', 0)
            put_text(f"Daily Avg: {daily:.1f} kWh", y_offset)

        # Combine
        if position == "right":
            combined = np.hstack([image, info])
        else:
            combined = np.hstack([info, image])

        return combined


class ChartGenerator:
    """Generates analysis charts using matplotlib."""

    @staticmethod
    def monthly_yield_chart(
        monthly_data: Dict[str, float],
        title: str = "Estimated Monthly Energy Yield",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Generate monthly yield bar chart."""
        months = list(monthly_data.keys())
        values = list(monthly_data.values())

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(months, values, color='#2196F3', edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Energy Yield (kWh)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.close()
            return None

    @staticmethod
    def financial_payback_chart(
        yearly_data: List[Dict],
        total_cost: float,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Generate cumulative profit / payback chart."""
        years = [d['year'] for d in yearly_data]
        cumulative = [d['cumulative_profit_myr'] for d in yearly_data]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years, cumulative, 'b-o', markersize=4, linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Breakeven')
        ax.fill_between(
            years, cumulative, 0,
            where=[c >= 0 for c in cumulative],
            alpha=0.3, color='green', label='Profit'
        )
        ax.fill_between(
            years, cumulative, 0,
            where=[c < 0 for c in cumulative],
            alpha=0.3, color='red', label='Investment Recovery'
        )

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Cumulative Profit (MYR)', fontsize=12)
        ax.set_title('25-Year Financial Projection', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.close()
            return None

    @staticmethod
    def training_history_chart(
        history: Dict,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Plot CNN training loss and accuracy curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Accuracy
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.close()
            return None

    @staticmethod
    def region_comparison_chart(
        region_data: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Compare energy yield across Malaysian regions."""
        regions = list(region_data.keys())
        yields = [region_data[r]['annual_yield_kwh'] for r in regions]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(regions, yields, color='#FF9800', edgecolor='white')

        ax.set_xlabel('Annual Energy Yield (kWh)', fontsize=12)
        ax.set_title('Regional Energy Yield Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for bar, val in zip(bars, yields):
            ax.text(val + 50, bar.get_y() + bar.get_height() / 2,
                    f'{val:,.0f}', va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.close()
            return None
