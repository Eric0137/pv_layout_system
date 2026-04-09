"""
Roof Boundary Detector - OpenCV-based
======================================
Detects usable roof area from uploaded images using classical computer vision.
Handles edge detection, contour extraction, and polygon approximation.
Users can define scale (pixels-to-meters) and north orientation.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from config import IMAGE_PROCESSING, LAYOUT


class RoofDetector:
    """
    Detects roof boundaries and usable area from uploaded roof images.
    Uses OpenCV for contour detection with adaptive thresholding.
    """

    def __init__(self):
        self.params = IMAGE_PROCESSING
        self.original_image = None
        self.processed_image = None
        self.roof_contour = None          # Main roof polygon (pixel coords)
        self.roof_polygon_m = None        # Roof polygon in meters
        self.usable_area_mask = None      # Binary mask of usable area
        self.scale_m_per_px = None        # Meters per pixel
        self.north_angle_deg = 0          # North direction (degrees from image top)
        self.image_shape = None

    def load_image(self, image_path: str) -> np.ndarray:
        """Load and validate roof image."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        self.original_image = img.copy()
        self.image_shape = img.shape[:2]
        return img

    def load_image_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """Load image from numpy array (for Streamlit uploads)."""
        self.original_image = image_array.copy()
        self.image_shape = image_array.shape[:2]
        return image_array

    def set_scale(self, known_length_m: float, pixel_length: float):
        """
        Set the scale factor from a known reference measurement.
        Args:
            known_length_m: Real-world length in meters
            pixel_length: Same length measured in pixels on the image
        """
        self.scale_m_per_px = known_length_m / pixel_length

    def set_scale_from_roof_dims(self, roof_length_m: float, roof_width_m: float):
        """
        Auto-calibrate scale using known roof dimensions.
        Matches the detected roof bounding box to the real dimensions.
        """
        if self.roof_contour is None:
            raise ValueError("Detect roof boundary first before setting scale.")
        rect = cv2.minAreaRect(self.roof_contour)
        (cx, cy), (w_px, h_px), angle = rect
        # Match longer side to longer dimension
        long_px = max(w_px, h_px)
        long_m = max(roof_length_m, roof_width_m)
        self.scale_m_per_px = long_m / long_px

    def set_north_angle(self, angle_deg: float):
        """Set north direction as degrees clockwise from image top."""
        self.north_angle_deg = angle_deg % 360

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for boundary detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            enhanced,
            (self.params["roof_detection_blur"], self.params["roof_detection_blur"]),
            0
        )
        return blurred

    def detect_roof_boundary(
        self,
        image: Optional[np.ndarray] = None,
        method: str = "auto"
    ) -> np.ndarray:
        """
        Detect the main roof boundary contour.

        Args:
            image: Input image (uses loaded image if None)
            method: Detection method - "auto", "canny", "adaptive", "color"

        Returns:
            Roof contour as numpy array of points
        """
        if image is None:
            image = self.original_image
        if image is None:
            raise ValueError("No image loaded.")

        preprocessed = self.preprocess(image)

        if method == "auto":
            # Try multiple methods and pick the best result
            contours = []
            for m in ["canny", "adaptive", "color"]:
                try:
                    c = self._detect_boundary_method(image, preprocessed, m)
                    if c is not None:
                        contours.append(c)
                except Exception:
                    continue

            if not contours:
                raise ValueError("Could not detect roof boundary with any method.")

            # Pick the contour with largest area
            self.roof_contour = max(contours, key=cv2.contourArea)
        else:
            self.roof_contour = self._detect_boundary_method(
                image, preprocessed, method
            )

        if self.roof_contour is None:
            raise ValueError(f"Could not detect roof boundary using {method}.")

        # Create usable area mask with edge setback
        self._create_usable_area_mask()
        return self.roof_contour

    def _detect_boundary_method(
        self,
        original: np.ndarray,
        preprocessed: np.ndarray,
        method: str
    ) -> Optional[np.ndarray]:
        """Apply a specific boundary detection method."""
        h, w = preprocessed.shape[:2]
        min_area = h * w * self.params["min_contour_area_ratio"]

        if method == "canny":
            edges = cv2.Canny(
                preprocessed,
                self.params["canny_low"],
                self.params["canny_high"]
            )
            # Dilate to close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.erode(edges, kernel, iterations=1)

        elif method == "adaptive":
            edges = cv2.adaptiveThreshold(
                preprocessed, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        elif method == "color":
            # Use color segmentation for colored roofs
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            # Detect dominant color region (likely the roof)
            h_channel = hsv[:, :, 0]
            hist = cv2.calcHist([h_channel], [0], None, [180], [0, 180])
            dominant_hue = np.argmax(hist)
            lower = np.array([max(0, dominant_hue - 20), 30, 30])
            upper = np.array([min(179, dominant_hue + 20), 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            edges = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            return None

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by minimum area
        valid = [c for c in contours if cv2.contourArea(c) > min_area]
        if not valid:
            return None

        # Get the largest contour
        largest = max(valid, key=cv2.contourArea)

        # Approximate to polygon for cleaner shape
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        return approx

    def _create_usable_area_mask(self):
        """Create a binary mask of usable roof area with edge setback."""
        h, w = self.image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.roof_contour], 255)

        # Apply edge setback if scale is known
        if self.scale_m_per_px is not None:
            setback_px = int(LAYOUT["min_edge_setback_m"] / self.scale_m_per_px)
            if setback_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * setback_px + 1, 2 * setback_px + 1)
                )
                mask = cv2.erode(mask, kernel, iterations=1)

        self.usable_area_mask = mask

    def get_roof_area_m2(self) -> float:
        """Calculate total roof area in square meters."""
        if self.roof_contour is None or self.scale_m_per_px is None:
            raise ValueError("Detect boundary and set scale first.")
        area_px = cv2.contourArea(self.roof_contour)
        return area_px * (self.scale_m_per_px ** 2)

    def get_usable_area_m2(self) -> float:
        """Calculate usable roof area (after setback) in square meters."""
        if self.usable_area_mask is None or self.scale_m_per_px is None:
            raise ValueError("Detect boundary and set scale first.")
        area_px = np.count_nonzero(self.usable_area_mask)
        return area_px * (self.scale_m_per_px ** 2)

    def get_roof_vertices_m(self) -> np.ndarray:
        """Get roof polygon vertices in meter coordinates."""
        if self.roof_contour is None or self.scale_m_per_px is None:
            raise ValueError("Detect boundary and set scale first.")
        vertices_px = self.roof_contour.reshape(-1, 2).astype(float)
        return vertices_px * self.scale_m_per_px

    def get_bounding_box_m(self) -> Tuple[float, float, float, float]:
        """Get roof bounding box in meters: (x, y, width, height)."""
        if self.roof_contour is None or self.scale_m_per_px is None:
            raise ValueError("Detect boundary and set scale first.")
        x, y, w, h = cv2.boundingRect(self.roof_contour)
        s = self.scale_m_per_px
        return (x * s, y * s, w * s, h * s)

    def remove_obstacles_from_mask(
        self,
        obstacle_contours: List[np.ndarray],
        buffer_m: Optional[float] = None
    ):
        """
        Remove detected obstacle regions from usable area mask.
        Args:
            obstacle_contours: List of obstacle contours in pixel coordinates
            buffer_m: Buffer/clearance around each obstacle (meters)
        """
        if self.usable_area_mask is None:
            raise ValueError("Run detect_roof_boundary first.")

        buffer_m = buffer_m or LAYOUT["obstacle_buffer_m"]

        for contour in obstacle_contours:
            if self.scale_m_per_px is not None:
                buffer_px = int(buffer_m / self.scale_m_per_px)
            else:
                buffer_px = 10  # Default pixel buffer

            # Create obstacle mask with buffer
            obstacle_mask = np.zeros_like(self.usable_area_mask)
            cv2.fillPoly(obstacle_mask, [contour], 255)

            if buffer_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * buffer_px + 1, 2 * buffer_px + 1)
                )
                obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)

            # Subtract from usable area
            self.usable_area_mask = cv2.bitwise_and(
                self.usable_area_mask,
                cv2.bitwise_not(obstacle_mask)
            )

    def detect_potential_obstacles(
        self,
        image: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Detect potential obstacle regions within the roof boundary.
        Returns candidate regions for CNN classification.
        """
        if image is None:
            image = self.original_image
        if self.roof_contour is None:
            raise ValueError("Detect roof boundary first.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Create roof-only image
        roof_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roof_mask, [self.roof_contour], 255)
        roof_gray = cv2.bitwise_and(gray, gray, mask=roof_mask)

        # Detect anomalies within roof area using multiple methods
        candidates = []

        # Method 1: Intensity-based anomaly detection
        mean_val = cv2.mean(roof_gray, mask=roof_mask)[0]
        std_val = np.std(roof_gray[roof_mask > 0])
        thresh_low = max(0, mean_val - 2 * std_val)
        thresh_high = min(255, mean_val + 2 * std_val)
        anomaly_mask = cv2.inRange(roof_gray, 0, int(thresh_low))
        anomaly_mask = cv2.bitwise_or(
            anomaly_mask,
            cv2.inRange(roof_gray, int(thresh_high), 255)
        )
        anomaly_mask = cv2.bitwise_and(anomaly_mask, roof_mask)

        # Method 2: Edge density detection
        edges = cv2.Canny(roof_gray, 80, 200)
        edges = cv2.bitwise_and(edges, roof_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        edge_density = cv2.dilate(edges, kernel, iterations=2)
        edge_density = cv2.erode(edge_density, kernel, iterations=1)

        # Combine methods
        combined = cv2.bitwise_or(anomaly_mask, edge_density)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

        # Find contours of potential obstacles
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = h * w * self.params["obstacle_min_area_ratio"]
        max_area = h * w * self.params["obstacle_max_area_ratio"]

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, bw, bh = cv2.boundingRect(contour)
                # Extract ROI for CNN classification
                roi = image[y:y+bh, x:x+bw]
                candidates.append({
                    "id": i,
                    "contour": contour,
                    "bbox": (x, y, bw, bh),
                    "area_px": area,
                    "roi": roi,
                    "label": "unknown",    # To be classified by CNN
                    "confidence": 0.0
                })

        return candidates

    def manually_define_roof(
        self,
        points_px: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Allow user to manually define roof boundary by clicking points.
        Args:
            points_px: List of (x, y) pixel coordinates defining roof polygon
        """
        self.roof_contour = np.array(points_px, dtype=np.int32).reshape(-1, 1, 2)
        self._create_usable_area_mask()
        return self.roof_contour

    def get_detection_summary(self) -> Dict:
        """Return a summary of detected roof properties."""
        summary = {
            "roof_detected": self.roof_contour is not None,
            "scale_set": self.scale_m_per_px is not None,
            "north_angle_deg": self.north_angle_deg,
            "num_vertices": len(self.roof_contour) if self.roof_contour is not None else 0,
        }
        if self.roof_contour is not None and self.scale_m_per_px is not None:
            summary["total_roof_area_m2"] = round(self.get_roof_area_m2(), 2)
            summary["usable_area_m2"] = round(self.get_usable_area_m2(), 2)
            bbox = self.get_bounding_box_m()
            summary["bounding_box_m"] = {
                "x": round(bbox[0], 2),
                "y": round(bbox[1], 2),
                "width": round(bbox[2], 2),
                "height": round(bbox[3], 2),
            }
        return summary
