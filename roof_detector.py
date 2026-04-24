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
        self.ridge_lines = []             # [[x1,y1],[x2,y2]] segments at roof peaks
        self.hip_lines = []               # [[x1,y1],[x2,y2]] diagonal hip segments
        self.valley_lines = []            # [[x1,y1],[x2,y2]] depression/valley segments

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
            method: Detection method - "auto", "grabcut", "canny",
                    "adaptive", "color"

        Returns:
            Roof contour as numpy array of points
        """
        if image is None:
            image = self.original_image
        if image is None:
            raise ValueError("No image loaded.")

        preprocessed = self.preprocess(image)

        if method == "auto":
            # Try multiple methods and pick the BEST-scoring result (not just
            # largest area, which can pick up background objects).
            scored = []
            for m in ["grabcut", "color", "adaptive", "canny"]:
                try:
                    c = self._detect_boundary_method(image, preprocessed, m)
                    if c is not None:
                        s = self._score_contour(c, image.shape[:2])
                        if s > 0:
                            scored.append((s, c, m))
                except Exception:
                    continue

            if not scored:
                raise ValueError("Could not detect roof boundary with any method.")

            scored.sort(key=lambda x: x[0], reverse=True)
            self.roof_contour = scored[0][1]
        else:
            self.roof_contour = self._detect_boundary_method(
                image, preprocessed, method
            )

        if self.roof_contour is None:
            raise ValueError(f"Could not detect roof boundary using {method}.")

        # Create usable area mask with edge setback
        self._create_usable_area_mask()
        return self.roof_contour

    # ------------------------------------------------------------------
    # DETECTION DISPATCHER
    # ------------------------------------------------------------------
    def _detect_boundary_method(
        self,
        original: np.ndarray,
        preprocessed: np.ndarray,
        method: str
    ) -> Optional[np.ndarray]:
        """Apply a specific boundary detection method.

        All methods now share a common post-processing pipeline:
          1. Produce a binary foreground mask.
          2. Fill holes and keep only the largest connected component.
          3. Fit a polygon using a fine-grained epsilon that actually
             follows the real roof edges.
        """
        h, w = original.shape[:2]
        img_area = float(h * w)
        min_area = img_area * self.params["min_contour_area_ratio"]
        max_area = img_area * 0.95  # avoid selecting the whole image

        if method == "grabcut":
            mask = self._method_grabcut(original)
        elif method == "color":
            mask = self._method_color_center(original)
        elif method == "canny":
            mask = self._method_canny(original, preprocessed)
        elif method == "adaptive":
            mask = self._method_adaptive(original, preprocessed)
        else:
            return None

        if mask is None or np.count_nonzero(mask) == 0:
            return None

        mask = self._postprocess_mask(mask)
        contour = self._largest_contour(mask, min_area, max_area)
        if contour is None:
            return None

        return self._refine_polygon(contour)

    # ------------------------------------------------------------------
    # INDIVIDUAL METHODS
    # ------------------------------------------------------------------
    def _method_grabcut(self, original: np.ndarray) -> Optional[np.ndarray]:
        """GrabCut with a center-rectangle prior.

        GrabCut models foreground (roof) vs background (trees, road, sky)
        with Gaussian Mixture Models in colour space. A center-rect
        initialization reliably separates the main roof from surroundings
        and is much more robust than simple colour thresholding when the
        roof's own colour is similar to nearby surfaces (e.g. parapets).
        """
        h, w = original.shape[:2]
        # Rectangle covering the central ~70% of the image. We assume the
        # user has framed the roof near the middle of the photo.
        mx = int(w * 0.12)
        my = int(h * 0.12)
        rect = (mx, my, max(1, w - 2 * mx), max(1, h - 2 * my))

        gc_mask = np.zeros((h, w), dtype=np.uint8)
        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(original, gc_mask, rect, bgd, fgd, 5,
                        cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return None

        fg = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        return fg

    def _method_color_center(self, original: np.ndarray) -> Optional[np.ndarray]:
        """Colour segmentation seeded from the IMAGE CENTRE in LAB space.

        Compared to the original "dominant-hue over whole image" approach
        this:
          - Samples colour statistics only from a central ROI (the roof is
            almost always near the image centre).
          - Uses the full LAB triplet (L, a, b) so it works for light,
            low-saturation roofs too (which pure-hue methods miss).
          - Uses a Mahalanobis-style normalised distance so the threshold
            adapts to how uniform the roof's colour actually is.
        """
        h, w = original.shape[:2]
        cy1, cy2 = int(h * 0.35), int(h * 0.65)
        cx1, cx2 = int(w * 0.35), int(w * 0.65)
        center = original[cy1:cy2, cx1:cx2]
        if center.size == 0:
            return None

        lab_full = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_center = cv2.cvtColor(center, cv2.COLOR_BGR2LAB).astype(np.float32)

        samples = lab_center.reshape(-1, 3)
        mean = samples.mean(axis=0)
        std = np.maximum(samples.std(axis=0), 8.0)  # floor to avoid div-by-0

        # Normalised distance from each pixel to the roof-centre colour
        norm = (lab_full - mean) / std
        dist = np.sqrt(np.sum(norm * norm, axis=2))

        # Adaptive threshold: start at 2 std and widen only if mask is empty
        for thr in (2.0, 2.5, 3.0, 3.5):
            mask = (dist < thr).astype(np.uint8) * 255
            if np.count_nonzero(mask) > 0.05 * h * w:
                break

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        return mask

    def _method_canny(
        self, original: np.ndarray, preprocessed: np.ndarray
    ) -> Optional[np.ndarray]:
        """Canny edges + flood-fill to turn edge map into a filled mask."""
        edges = cv2.Canny(
            preprocessed,
            self.params["canny_low"],
            self.params["canny_high"]
        )
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, k, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

        # Invert so edges act as walls; flood-fill from image corners to
        # mark background; then foreground = everything else.
        inv = cv2.bitwise_not(edges)
        h, w = inv.shape
        ff = inv.copy()
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            if ff[seed[1], seed[0]] == 255:
                cv2.floodFill(ff, ff_mask, seed, 0)

        # ff now has foreground (non-background) regions as 255
        mask = ff
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=2)
        return mask

    def _method_adaptive(
        self, original: np.ndarray, preprocessed: np.ndarray
    ) -> Optional[np.ndarray]:
        """Adaptive threshold with larger block size for smoother regions."""
        th = cv2.adaptiveThreshold(
            preprocessed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        return mask

    # ------------------------------------------------------------------
    # POST-PROCESSING & REFINEMENT
    # ------------------------------------------------------------------
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Fill holes, smooth boundary, keep only the largest blob."""
        # Fill holes by drawing all external contours as filled polygons.
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return mask
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

        # Keep only the largest connected component.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            filled, connectivity=8
        )
        if num_labels <= 1:
            return filled
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return filled
        largest_label = 1 + int(np.argmax(areas))
        largest = np.where(labels == largest_label, 255, 0).astype(np.uint8)

        # Gentle erosion to shave off thin parapet fringes that share the
        # roof's colour. ~1% of the image min-dimension.
        shave_px = max(1, int(min(mask.shape[:2]) * 0.008))
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * shave_px + 1, 2 * shave_px + 1)
        )
        shaved = cv2.erode(largest, k, iterations=1)
        # Guard against erosion wiping everything out for small roofs.
        if np.count_nonzero(shaved) > 0.2 * np.count_nonzero(largest):
            largest = cv2.dilate(shaved, k, iterations=1)

        return largest

    def _largest_contour(
        self, mask: np.ndarray, min_area: float, max_area: float
    ) -> Optional[np.ndarray]:
        """Pull the largest in-range contour out of a binary mask."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        valid = [
            c for c in contours
            if min_area < cv2.contourArea(c) < max_area
        ]
        if not valid:
            return None
        return max(valid, key=cv2.contourArea)

    def _refine_polygon(self, contour: np.ndarray) -> np.ndarray:
        """Produce a polygon that closely follows the real roof edges.

        Uses a small initial epsilon (0.003 * perimeter) and only grows it
        until the vertex count is reasonable (~6-30). This avoids both the
        ultra-coarse 4-point quad the old code sometimes produced, and
        hundreds of jittery micro-vertices.
        """
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return contour

        epsilon = 0.003 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Cap maximum vertex count for a clean visual.
        max_vertices = 30
        guard = 0
        while len(approx) > max_vertices and guard < 12:
            epsilon *= 1.3
            approx = cv2.approxPolyDP(contour, epsilon, True)
            guard += 1

        # Guarantee at least a triangle.
        if len(approx) < 3:
            approx = cv2.approxPolyDP(contour, 0.001 * perimeter, True)

        return approx

    def _score_contour(
        self, contour: np.ndarray, image_shape: Tuple[int, int]
    ) -> float:
        """Score a contour by area + centrality + convexity.

        Returns a number in [0, 1]. Used by "auto" method to pick the
        candidate most likely to be the actual roof.
        """
        h, w = image_shape
        img_area = float(h * w)
        area = cv2.contourArea(contour)
        if area <= 0:
            return -1.0

        # Area: roofs usually occupy 15-85% of a well-framed photo.
        area_ratio = area / img_area
        if 0.15 <= area_ratio <= 0.85:
            area_score = 1.0
        elif area_ratio < 0.15:
            area_score = area_ratio / 0.15
        else:
            area_score = max(0.0, (1.0 - area_ratio) / 0.15)

        # Centrality: centroid close to image centre.
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return -1.0
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        max_d = np.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)
        center_d = np.sqrt((cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2)
        centrality = 1.0 - (center_d / max_d)

        # Convexity (solidity): roofs are mostly convex-ish.
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = (area / hull_area) if hull_area > 0 else 0.0

        return 0.45 * area_score + 0.35 * centrality + 0.20 * convexity

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

    # ------------------------------------------------------------------
    # RIDGE / HIP / VALLEY LINE DETECTION
    # ------------------------------------------------------------------

    def detect_roof_lines(
        self,
        image: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Detect ridge, hip, and valley (depression) lines within the roof.

        Ridge: bright peak where two upward slopes meet at the top.
        Hip:   diagonal bright line from a ridge end down to a roof corner.
        Valley: dark depression where two inward slopes meet (roof joins).

        Requires detect_roof_boundary() to have been called first.

        Returns:
            Dict with keys 'ridges', 'hips', 'valleys' — each a list of
            [[x1,y1],[x2,y2]] integer pixel-coordinate segments.
        """
        if image is None:
            image = self.original_image
        if self.roof_contour is None:
            raise ValueError("Detect roof boundary first.")

        h, w = image.shape[:2]
        roof_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roof_mask, [self.roof_contour], 255)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_full = clahe.apply(gray)
        enhanced = np.zeros_like(gray)
        enhanced[roof_mask > 0] = enhanced_full[roof_mask > 0]

        candidates = self._detect_structural_lines(enhanced, roof_mask)

        if candidates is None or len(candidates) == 0:
            self.ridge_lines = []
            self.hip_lines = []
            self.valley_lines = []
            return {"ridges": [], "hips": [], "valleys": []}

        principal_angle = self._get_roof_principal_angle()
        min_len = max(15.0, min(h, w) * 0.04)

        ridges, hips, valleys = [], [], []

        for seg in candidates:
            x1, y1, x2, y2 = int(seg[0][0]), int(seg[0][1]), int(seg[0][2]), int(seg[0][3])
            length = np.hypot(x2 - x1, y2 - y1)
            if length < min_len:
                continue

            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            if not (0 <= mid_y < h and 0 <= mid_x < w):
                continue
            if roof_mask[mid_y, mid_x] == 0:
                continue

            binfo = self._analyze_line_brightness(enhanced, roof_mask, x1, y1, x2, y2)
            angle = float(np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))))
            label = self._classify_roof_line(angle, binfo, principal_angle, x1, y1, x2, y2)

            if label == "ridge":
                ridges.append([[x1, y1], [x2, y2]])
            elif label == "hip":
                hips.append([[x1, y1], [x2, y2]])
            elif label == "valley":
                valleys.append([[x1, y1], [x2, y2]])

        self.ridge_lines = self._merge_line_segments(ridges, angle_tol=15.0, dist_tol=25.0)
        self.hip_lines = self._merge_line_segments(hips, angle_tol=20.0, dist_tol=25.0)
        self.valley_lines = self._merge_line_segments(valleys, angle_tol=15.0, dist_tol=20.0)

        return {
            "ridges": self.ridge_lines,
            "hips": self.hip_lines,
            "valleys": self.valley_lines,
        }

    def _detect_structural_lines(
        self, enhanced: np.ndarray, roof_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract candidate structural line segments via Probabilistic Hough."""
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        edges = cv2.Canny(blurred, 25, 75)
        edges = cv2.bitwise_and(edges, roof_mask)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, k, iterations=1)

        h, w = roof_mask.shape
        min_len = int(max(20, min(h, w) * 0.05))
        return cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,
            threshold=30,
            minLineLength=min_len,
            maxLineGap=15,
        )

    def _get_roof_principal_angle(self) -> float:
        """Return the primary orientation of the roof bounding rectangle in [0, 90]°."""
        rect = cv2.minAreaRect(self.roof_contour)
        angle = abs(rect[2])
        if angle > 45:
            angle = 90.0 - angle
        return angle

    def _analyze_line_brightness(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        sample_width: int = 15,
    ) -> Dict:
        """
        Sample pixels perpendicular to the line segment to determine whether it
        sits at a brightness peak (ridge/hip) or a brightness depression (valley).
        """
        h, w = gray.shape
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = np.hypot(dx, dy)
        if length < 1.0:
            return {"is_bright_peak": False, "is_dark_valley": False, "contrast": 0.0}

        ux, uy = dx / length, dy / length
        px, py = -uy, ux  # left-perpendicular unit vector

        on_vals: List[float] = []
        left_vals: List[float] = []
        right_vals: List[float] = []
        num_samples = max(5, int(length / 8))

        for i in range(num_samples):
            t = i / max(1, num_samples - 1)
            cx = int(round(x1 + t * dx))
            cy = int(round(y1 + t * dy))

            if 0 <= cy < h and 0 <= cx < w and mask[cy, cx] > 0:
                on_vals.append(float(gray[cy, cx]))

            for sign, vals in ((1.0, left_vals), (-1.0, right_vals)):
                for off in range(4, sample_width, 4):
                    sx = int(round(cx + sign * px * off))
                    sy = int(round(cy + sign * py * off))
                    if 0 <= sy < h and 0 <= sx < w and mask[sy, sx] > 0:
                        vals.append(float(gray[sy, sx]))

        if not on_vals:
            return {"is_bright_peak": False, "is_dark_valley": False, "contrast": 0.0}

        on_mean = float(np.mean(on_vals))
        left_mean = float(np.mean(left_vals)) if left_vals else on_mean
        right_mean = float(np.mean(right_vals)) if right_vals else on_mean
        thr = 6.0

        return {
            "on_mean": on_mean,
            "left_mean": left_mean,
            "right_mean": right_mean,
            "is_bright_peak": on_mean > left_mean + thr and on_mean > right_mean + thr,
            "is_dark_valley": on_mean < left_mean - thr and on_mean < right_mean - thr,
            "contrast": on_mean - (left_mean + right_mean) / 2.0,
        }

    def _classify_roof_line(
        self,
        angle: float,
        binfo: Dict,
        principal_angle: float,
        x1: int, y1: int, x2: int, y2: int,
    ) -> Optional[str]:
        """
        Classify a candidate line as 'ridge', 'hip', 'valley', or None.

        Ridge: bright peak, roughly parallel to the roof principal axis (±25°).
        Hip:   bright peak, at significant angle to principal axis, at least one
               endpoint within 20 px of the roof boundary.
        Valley: dark depression anywhere within the roof.
        """
        if binfo["is_dark_valley"]:
            return "valley"

        if binfo["is_bright_peak"]:
            da = abs(angle - principal_angle)
            if da > 45:
                da = 90.0 - da
            if da <= 25:
                return "ridge"
            # Diagonal bright line — classify as hip if endpoint touches boundary
            boundary_thr = max(20, int(min(self.image_shape) * 0.04))
            if (self._endpoint_near_boundary(x1, y1, boundary_thr) or
                    self._endpoint_near_boundary(x2, y2, boundary_thr)):
                return "hip"
            return "ridge"  # interior diagonal without boundary contact → ridge

        return None

    def _endpoint_near_boundary(self, x: int, y: int, threshold: int) -> bool:
        """Return True if (x, y) is within threshold pixels of any roof contour point."""
        pts = self.roof_contour.reshape(-1, 2).astype(float)
        dists = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        return bool(np.min(dists) <= threshold)

    def _merge_line_segments(
        self,
        segments: List,
        angle_tol: float = 15.0,
        dist_tol: float = 25.0,
    ) -> List:
        """
        Merge nearly collinear, spatially close line segments into single segments.
        Groups by similar angle and endpoint proximity, then spans the full extent.
        """
        if len(segments) <= 1:
            return segments

        segs = np.array(segments, dtype=float)  # (N, 2, 2)
        angles = np.degrees(np.arctan2(
            segs[:, 1, 1] - segs[:, 0, 1],
            segs[:, 1, 0] - segs[:, 0, 0],
        )) % 180.0

        used = [False] * len(segs)
        merged: List = []

        for i in range(len(segs)):
            if used[i]:
                continue
            group = [i]
            used[i] = True

            for j in range(i + 1, len(segs)):
                if used[j]:
                    continue
                da = abs(angles[i] - angles[j])
                if da > 90:
                    da = 180.0 - da
                if da > angle_tol:
                    continue
                # Close enough if any endpoint pair is within dist_tol
                pts_i, pts_j = segs[i], segs[j]
                dists = [
                    np.hypot(pts_i[0, 0] - pts_j[0, 0], pts_i[0, 1] - pts_j[0, 1]),
                    np.hypot(pts_i[0, 0] - pts_j[1, 0], pts_i[0, 1] - pts_j[1, 1]),
                    np.hypot(pts_i[1, 0] - pts_j[0, 0], pts_i[1, 1] - pts_j[0, 1]),
                    np.hypot(pts_i[1, 0] - pts_j[1, 0], pts_i[1, 1] - pts_j[1, 1]),
                ]
                if min(dists) <= dist_tol:
                    group.append(j)
                    used[j] = True

            # Span the group: pick the two farthest endpoints
            all_pts = np.vstack([segs[k] for k in group])  # (2*N, 2)
            max_d, best = -1.0, (all_pts[0], all_pts[1])
            for a in range(len(all_pts)):
                for b in range(a + 1, len(all_pts)):
                    d = np.hypot(all_pts[a, 0] - all_pts[b, 0],
                                 all_pts[a, 1] - all_pts[b, 1])
                    if d > max_d:
                        max_d, best = d, (all_pts[a], all_pts[b])

            p1, p2 = best
            merged.append([[int(p1[0]), int(p1[1])], [int(p2[0]), int(p2[1])]])

        return merged

    def get_roof_lines_m(self) -> Dict:
        """Return detected ridge/hip/valley lines converted to meter coordinates."""
        if self.scale_m_per_px is None:
            raise ValueError("Set scale first.")
        s = self.scale_m_per_px

        def to_m(segs: List) -> List:
            return [
                [[pt[0] * s, pt[1] * s] for pt in seg]
                for seg in segs
            ]

        return {
            "ridges": to_m(self.ridge_lines),
            "hips": to_m(self.hip_lines),
            "valleys": to_m(self.valley_lines),
        }

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
            "ridge_lines": len(self.ridge_lines),
            "hip_lines": len(self.hip_lines),
            "valley_lines": len(self.valley_lines),
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
