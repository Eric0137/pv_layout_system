"""
Obstacle Classifier - Trainable CNN
====================================
A lightweight CNN model for classifying detected roof obstacles.
Supports training from user-provided datasets with data augmentation.
Classes: water_tank, parapet, vent, ac_unit, none (not an obstacle)
"""

import os
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
from config import CNN_CONFIG, IMAGE_PROCESSING

# ============================================================
# PyTorch imports (with graceful fallback)
# ============================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not installed. CNN classifier unavailable.")
    print("Install with: pip install torch torchvision")


# ============================================================
# Custom Dataset
# ============================================================
class ObstacleDataset:
    """Dataset loader for obstacle classification images."""

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir: Path to dataset/obstacles/ directory with class subfolders
            transform: Optional torchvision transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = CNN_CONFIG["class_names"]
        self.samples = []  # List of (image_path, class_index)

        # Scan class directories
        for idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                os.makedirs(class_dir, exist_ok=True)
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append(
                        (os.path.join(class_dir, fname), idx)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            # Return a blank image if file is corrupted
            size = CNN_CONFIG.get("input_size", IMAGE_PROCESSING["cnn_input_size"])
            image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_PROCESSING["cnn_input_size"])

        if TORCH_AVAILABLE and self.transform:
            image = self.transform(image)
        else:
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            if TORCH_AVAILABLE:
                image = torch.from_numpy(image)

        if TORCH_AVAILABLE:
            label = torch.tensor(label, dtype=torch.long)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of samples per class."""
        dist = {name: 0 for name in self.class_names}
        for _, idx in self.samples:
            dist[self.class_names[idx]] += 1
        return dist


# ============================================================
# CNN Architecture
# ============================================================
if TORCH_AVAILABLE:
    class ObstacleCNN(nn.Module):
        """
        Lightweight CNN for obstacle classification.
        Architecture: 4 conv blocks + global average pooling + FC.
        Designed for small datasets with strong regularization.
        """

        def __init__(self, num_classes: int = CNN_CONFIG["num_classes"]):
            super(ObstacleCNN, self).__init__()

            self.features = nn.Sequential(
                # Block 1: 3 -> 32 channels
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),

                # Block 2: 32 -> 64 channels
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),

                # Block 3: 64 -> 128 channels
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),

                # Block 4: 128 -> 256 channels
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
            )

            # Global Average Pooling
            self.gap = nn.AdaptiveAvgPool2d(1)

            # Classifier
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.gap(x)
            x = self.classifier(x)
            return x


# ============================================================
# Classifier Wrapper (High-level API)
# ============================================================
class ObstacleClassifier:
    """
    High-level API for training and using the obstacle classifier.
    Handles data loading, augmentation, training, and inference.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.class_names = CNN_CONFIG["class_names"]
        self.is_trained = False
        self.training_history = {"train_loss": [], "val_loss": [],
                                 "train_acc": [], "val_acc": []}
        self.train_transform = None
        self.val_transform = None

        if TORCH_AVAILABLE:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.model = ObstacleCNN(CNN_CONFIG["num_classes"]).to(self.device)
            # Data augmentation transforms for training
            self._setup_transforms()

    def _setup_transforms(self):
        """Setup data augmentation pipelines."""
        if not TORCH_AVAILABLE:
            return

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def prepare_datasets(
        self,
        dataset_dir: str
    ) -> tuple:
        """
        Prepare train/val/test dataloaders from dataset directory.
        Args:
            dataset_dir: Path to obstacles directory with class subfolders
        Returns:
            (train_loader, val_loader, test_loader)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training.")

        # Load full dataset
        full_dataset = ObstacleDataset(dataset_dir, transform=None)
        print(f"[INFO] Dataset loaded: {len(full_dataset)} samples")
        print(f"[INFO] Class distribution: {full_dataset.get_class_distribution()}")

        if len(full_dataset) == 0:
            raise ValueError(
                f"No images found in {dataset_dir}. "
                f"Add images to class subfolders: {self.class_names}"
            )

        # Split dataset
        n = len(full_dataset)
        n_train = int(n * CNN_CONFIG["train_split"])
        n_val = int(n * CNN_CONFIG["val_split"])
        n_test = n - n_train - n_val

        # Create index splits
        indices = list(range(n))
        np.random.shuffle(indices)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        # Create separate datasets with appropriate transforms
        train_dataset = ObstacleDataset(dataset_dir, transform=self.train_transform)
        val_dataset = ObstacleDataset(dataset_dir, transform=self.val_transform)
        test_dataset = ObstacleDataset(dataset_dir, transform=self.val_transform)

        # Use subset samplers
        from torch.utils.data import SubsetRandomSampler
        train_loader = DataLoader(
            train_dataset,
            batch_size=CNN_CONFIG["batch_size"],
            sampler=SubsetRandomSampler(train_idx)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=CNN_CONFIG["batch_size"],
            sampler=SubsetRandomSampler(val_idx)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CNN_CONFIG["batch_size"],
            sampler=SubsetRandomSampler(test_idx)
        )

        return train_loader, val_loader, test_loader

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = CNN_CONFIG["epochs"],
        lr: float = CNN_CONFIG["learning_rate"]
    ) -> Dict:
        """
        Train the obstacle classifier.
        Returns training history dict.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')
        patience_counter = 0
        self.training_history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": []
        }

        print(f"\n[TRAINING] Device: {self.device}")
        print(f"[TRAINING] Epochs: {epochs}, LR: {lr}")
        print("-" * 60)

        for epoch in range(epochs):
            # --- Training Phase ---
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            # Update scheduler
            scheduler.step(val_loss)

            # Record history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["train_acc"].append(train_acc)
            self.training_history["val_acc"].append(val_acc)

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
                )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(CNN_CONFIG["model_save_path"])
            else:
                patience_counter += 1
                if patience_counter >= CNN_CONFIG["early_stopping_patience"]:
                    print(f"\n[EARLY STOPPING] at epoch {epoch+1}")
                    break

        self.is_trained = True
        print("-" * 60)
        print(f"[TRAINING COMPLETE] Best Val Loss: {best_val_loss:.4f}")
        return self.training_history

    def evaluate(self, test_loader) -> Dict:
        """Evaluate model on test set. Returns metrics dict."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required.")

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Per-class accuracy
        class_acc = {}
        for i, name in enumerate(self.class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                class_acc[name] = (all_preds[mask] == all_labels[mask]).mean()

        overall_acc = (all_preds == all_labels).mean()

        return {
            "overall_accuracy": float(overall_acc),
            "class_accuracy": class_acc,
            "predictions": all_preds.tolist(),
            "ground_truth": all_labels.tolist(),
        }

    def classify(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single obstacle ROI image.
        Args:
            image: BGR image array of the obstacle region
        Returns:
            (class_name, confidence)
        """
        if not TORCH_AVAILABLE or self.model is None:
            return "unknown", 0.0

        if not self.is_trained:
            return "unknown", 0.0

        # Preprocess
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_PROCESSING["cnn_input_size"])

        if self.val_transform:
            img_tensor = self.val_transform(img).unsqueeze(0).to(self.device)
        else:
            img_tensor = torch.from_numpy(
                img.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        class_name = self.class_names[predicted.item()]
        return class_name, confidence.item()

    def classify_batch(
        self, obstacles: List[Dict]
    ) -> List[Dict]:
        """
        Classify a list of obstacle candidates from RoofDetector.
        Updates each obstacle dict with label and confidence.
        """
        for obs in obstacles:
            if obs["roi"] is not None and obs["roi"].size > 0:
                label, conf = self.classify(obs["roi"])
                obs["label"] = label
                obs["confidence"] = conf
            else:
                obs["label"] = "unknown"
                obs["confidence"] = 0.0
        return obstacles

    def save_model(self, path: str):
        """Save model weights."""
        if TORCH_AVAILABLE and self.model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'class_names': self.class_names,
                'is_trained': self.is_trained,
            }, path)
            print(f"[SAVED] Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint.get('class_names', self.class_names)
        self.is_trained = checkpoint.get('is_trained', True)
        self.model.eval()
        print(f"[LOADED] Model loaded from {path}")


# ============================================================
# Synthetic Data Generator (for bootstrapping training)
# ============================================================
class SyntheticObstacleGenerator:
    """
    Generates synthetic training images for obstacle classification.
    Useful for bootstrapping when no real dataset is available yet.
    """

    @staticmethod
    def generate_water_tank(size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """Generate synthetic water tank image."""
        img = np.random.randint(180, 220, (*size, 3), dtype=np.uint8)
        h, w = size
        # Draw cylindrical tank shape
        cv2.ellipse(img, (w//2, h//3), (w//3, h//6), 0, 0, 360, (100, 100, 110), -1)
        cv2.rectangle(img, (w//2 - w//3, h//3), (w//2 + w//3, 2*h//3), (100, 100, 110), -1)
        cv2.ellipse(img, (w//2, 2*h//3), (w//3, h//6), 0, 0, 360, (90, 90, 100), -1)
        # Add some noise
        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def generate_vent(size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """Generate synthetic vent/exhaust image."""
        img = np.random.randint(160, 200, (*size, 3), dtype=np.uint8)
        h, w = size
        # Draw circular vent with slats
        cv2.circle(img, (w//2, h//2), w//3, (80, 80, 85), -1)
        for i in range(-3, 4):
            y = h//2 + i * (w//3) // 4
            cv2.line(img, (w//2 - w//4, y), (w//2 + w//4, y), (60, 60, 65), 2)
        noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def generate_ac_unit(size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """Generate synthetic AC unit image."""
        img = np.random.randint(170, 210, (*size, 3), dtype=np.uint8)
        h, w = size
        # Draw rectangular box with fan grid
        cv2.rectangle(img, (w//6, h//4), (5*w//6, 3*h//4), (200, 200, 210), -1)
        cv2.rectangle(img, (w//6, h//4), (5*w//6, 3*h//4), (120, 120, 130), 2)
        # Fan circle
        cv2.circle(img, (w//2, h//2), w//5, (150, 150, 160), 2)
        for angle in range(0, 360, 45):
            rad = np.radians(angle)
            x1 = int(w//2 + (w//5 - 5) * np.cos(rad))
            y1 = int(h//2 + (w//5 - 5) * np.sin(rad))
            cv2.line(img, (w//2, h//2), (x1, y1), (140, 140, 150), 1)
        noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def generate_parapet(size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """Generate synthetic parapet wall image."""
        img = np.random.randint(150, 190, (*size, 3), dtype=np.uint8)
        h, w = size
        # Draw wall-like structure
        cv2.rectangle(img, (0, h//3), (w, 2*h//3), (130, 125, 120), -1)
        # Add brick-like texture
        for row in range(h//3, 2*h//3, 12):
            cv2.line(img, (0, row), (w, row), (110, 105, 100), 1)
            offset = 0 if (row // 12) % 2 == 0 else 20
            for col in range(offset, w, 40):
                cv2.line(img, (col, row), (col, min(row + 12, 2*h//3)), (110, 105, 100), 1)
        noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def generate_none(size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """Generate image of empty roof area (no obstacle)."""
        # Random roof-like color
        base_color = np.random.randint(130, 200, 3)
        img = np.full((*size, 3), base_color, dtype=np.uint8)
        # Add some texture
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    def generate_dataset(
        self,
        output_dir: str,
        samples_per_class: int = 200,
        size: Tuple[int, int] = (128, 128)
    ):
        """
        Generate a full synthetic dataset for initial training.
        Args:
            output_dir: Path to obstacles directory
            samples_per_class: Number of images per class
            size: Image size (height, width)
        """
        generators = {
            "water_tank": self.generate_water_tank,
            "parapet": self.generate_parapet,
            "vent": self.generate_vent,
            "ac_unit": self.generate_ac_unit,
            "none": self.generate_none,
        }

        total = 0
        for class_name, gen_func in generators.items():
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            for i in range(samples_per_class):
                img = gen_func(size)
                # Apply random augmentations for variety
                if np.random.random() > 0.5:
                    img = cv2.flip(img, 1)  # Horizontal flip
                if np.random.random() > 0.5:
                    img = cv2.flip(img, 0)  # Vertical flip
                if np.random.random() > 0.3:
                    angle = np.random.uniform(-15, 15)
                    M = cv2.getRotationMatrix2D(
                        (size[1]//2, size[0]//2), angle, 1.0
                    )
                    img = cv2.warpAffine(img, M, (size[1], size[0]))

                fname = f"{class_name}_{i:04d}.png"
                cv2.imwrite(os.path.join(class_dir, fname), img)
                total += 1

        print(f"[GENERATED] {total} synthetic images in {output_dir}")
        print(f"[INFO] {samples_per_class} per class x {len(generators)} classes")
