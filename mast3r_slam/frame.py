import dataclasses
from enum import Enum
from typing import Optional
import lietorch
import torch
from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.config import config
import pathlib
import numpy as np


# Global variable to store loaded intrinsics
_loaded_intrinsics = None


def load_intrinsics_file(intrinsics_path):
    """Load camera intrinsics from a simple text file.
    
    Expected format:
    Line 1: fx, fy, ox, oy (header - ignored)
    Line 2: focal_x, focal_y, principal_x, principal_y (values)
    
    Args:
        intrinsics_path: Path to the intrinsics file
        
    Returns:
        dict: Dictionary with intrinsics values or None if file doesn't exist
    """
    global _loaded_intrinsics
    
    if _loaded_intrinsics is not None:
        return _loaded_intrinsics
        
    if intrinsics_path is None:
        return None
        
    intrinsics_file = pathlib.Path(intrinsics_path)
    if not intrinsics_file.exists():
        print(f"Warning: Intrinsics file {intrinsics_path} not found")
        return None
    
    try:
        with open(intrinsics_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            print(f"Warning: Intrinsics file {intrinsics_path} has insufficient lines")
            return None
            
        # Parse the second line which contains the values
        values_line = lines[1].strip()
        values = [float(x.strip()) for x in values_line.split(',')]
        
        if len(values) != 4:
            print(f"Warning: Expected 4 intrinsics values, got {len(values)}")
            return None
            
        fx, fy, ox, oy = values
        
        _loaded_intrinsics = {
            'fx': fx,
            'fy': fy,
            'ox': ox,
            'oy': oy
        }
        
        print(f"Loaded intrinsics from {intrinsics_path}:")
        print(f"  fx={fx:.4f}, fy={fy:.4f}")
        print(f"  ox={ox:.4f}, oy={oy:.4f}")
        
        return _loaded_intrinsics
        
    except Exception as e:
        print(f"Error loading intrinsics file {intrinsics_path}: {e}")
        return None


class Mode(Enum):
    INIT = 0
    TRACKING = 1
    RELOC = 2
    TERMINATED = 3


@dataclasses.dataclass
class Frame:
    frame_id: int
    img: torch.Tensor
    img_shape: torch.Tensor
    img_true_shape: torch.Tensor
    uimg: torch.Tensor
    T_WC: lietorch.Sim3 = lietorch.Sim3.Identity(1)
    X_canon: Optional[torch.Tensor] = None
    C: Optional[torch.Tensor] = None
    feat: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None
    N: int = 0
    N_updates: int = 0
    K: Optional[torch.Tensor] = None
    dense_depth: Optional[torch.Tensor] = None  # Store original dense depth from MASt3R

    def get_score(self, C):
        filtering_score = config["tracking"]["filtering_score"]
        if filtering_score == "median":
            score = torch.median(C)  # Is this slower than mean? Is it worth it?
        elif filtering_score == "mean":
            score = torch.mean(C)
        return score

    def update_pointmap(self, X: torch.Tensor, C: torch.Tensor):
        filtering_mode = config["tracking"]["filtering_mode"]

        if self.N == 0:
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
            self.N_updates = 1
            if filtering_mode == "best_score":
                self.score = self.get_score(C)
            return

        if filtering_mode == "first":
            if self.N_updates == 1:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
        elif filtering_mode == "recent":
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
        elif filtering_mode == "best_score":
            new_score = self.get_score(C)
            if new_score > self.score:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
                self.score = new_score
        elif filtering_mode == "indep_conf":
            new_mask = C > self.C
            self.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
            self.C[new_mask] = C[new_mask]
            self.N = 1
        elif filtering_mode == "weighted_pointmap":
            self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
            self.C = self.C + C
            self.N += 1
        elif filtering_mode == "weighted_spherical":

            def cartesian_to_spherical(P):
                r = torch.linalg.norm(P, dim=-1, keepdim=True)
                x, y, z = torch.tensor_split(P, 3, dim=-1)
                phi = torch.atan2(y, x)
                theta = torch.acos(z / r)
                spherical = torch.cat((r, phi, theta), dim=-1)
                return spherical

            def spherical_to_cartesian(spherical):
                r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
                x = r * torch.sin(theta) * torch.cos(phi)
                y = r * torch.sin(theta) * torch.sin(phi)
                z = r * torch.cos(theta)
                P = torch.cat((x, y, z), dim=-1)
                return P

            spherical1 = cartesian_to_spherical(self.X_canon)
            spherical2 = cartesian_to_spherical(X)
            spherical = ((self.C * spherical1) + (C * spherical2)) / (self.C + C)

            self.X_canon = spherical_to_cartesian(spherical)
            self.C = self.C + C
            self.N += 1

        self.N_updates += 1
        return

    def get_average_conf(self):
        return self.C / self.N if self.C is not None else None

    def get_intrinsics_string(self, dataset_intrinsics=None):
        """Get camera intrinsics as a string in the format: width height fx fy cx cy k1 k2 p1 p2 k3
        
        Returns:
            str: Intrinsics string, or estimated intrinsics if no calibration available
        """
        # Get image dimensions first (needed for all cases)
        img_shape_flat = self.img_shape.flatten()
        height, width = int(img_shape_flat[0].item()), int(img_shape_flat[1].item())
        
        if self.K is None:
            # First, try to load intrinsics from file if specified in config
            intrinsics_path = config.get("intrinsics_file")
            loaded_intrinsics = load_intrinsics_file(intrinsics_path)
            
            if loaded_intrinsics is not None:
                # Use intrinsics loaded from file
                fx = loaded_intrinsics['fx']
                fy = loaded_intrinsics['fy']
                cx = loaded_intrinsics['ox']  # ox = principal point x
                cy = loaded_intrinsics['oy']  # oy = principal point y
                k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0  # No distortion
                
                print(f"Using loaded camera intrinsics:")
                print(f"  fx={fx:.4f}, fy={fy:.4f}")
                print(f"  cx={cx:.4f}, cy={cy:.4f}")
                
                return f"{width} {height} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2} {k3}"
            
            # Fallback: Estimate reasonable intrinsics based on image size
            # Based on real smartphone camera measurements (iPhone data shows fx/width ≈ 0.8-0.9)
            # This gives focal_length ≈ 0.85 * max(width, height) for typical smartphone cameras
            
            # Use smartphone-calibrated focal length estimate
            # Based on iPhone camera specs: fx ≈ 0.8-0.9 × max_dimension
            max_dim = max(width, height)
            fx = fy = max_dim * 0.85  # Conservative estimate for smartphone cameras
            
            cx, cy = width / 2.0, height / 2.0  # Principal point at image center
            k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0  # No distortion
            
            print(f"Estimated camera intrinsics for {width}x{height} image:")
            print(f"  fx=fy={fx:.1f} (based on smartphone camera specs)")
            print(f"  fx/max_dim ratio: {fx/max_dim:.2f}")
            print(f"  cx={cx:.1f}, cy={cy:.1f}")
            
            return f"{width} {height} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2} {k3}"
        
        # Get focal lengths and principal point from calibrated intrinsics
        fx = float(self.K[0, 0])
        fy = float(self.K[1, 1])
        cx = float(self.K[0, 2])
        cy = float(self.K[1, 2])
        
        # Get distortion coefficients from dataset intrinsics if available
        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0
        if dataset_intrinsics is not None and hasattr(dataset_intrinsics, 'distortion'):
            distortion = dataset_intrinsics.distortion
            if distortion is not None and len(distortion) >= 4:
                k1 = float(distortion[0]) if len(distortion) > 0 else 0.0
                k2 = float(distortion[1]) if len(distortion) > 1 else 0.0
                p1 = float(distortion[2]) if len(distortion) > 2 else 0.0
                p2 = float(distortion[3]) if len(distortion) > 3 else 0.0
                k3 = float(distortion[4]) if len(distortion) > 4 else 0.0
            
        return f"{width} {height} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2} {k3}"

    def save_depth_map(self, output_dir):
        """Save depth map to file and return the path"""
        if self.dense_depth is not None:
            # Use original dense depth map from MASt3R (preferred)
            output_dir = pathlib.Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract depth as Euclidean distance from camera center
            # This handles scale inconsistency better than raw Z-coordinates
            depth_map = np.linalg.norm(self.dense_depth.cpu().numpy(), axis=2)
            
            # Filter out invalid depths using adaptive threshold
            if depth_map.max() > 0:
                # Use a threshold relative to the scene scale
                threshold = depth_map.max() * 1e-4  # 0.01% of max depth
                depth_map[depth_map <= threshold] = 0.0
            else:
                depth_map[depth_map <= 0] = 0.0
            
            # Save depth map with consistent frame_id naming (no "depth_" prefix)
            depth_path = output_dir / f"{self.frame_id:06d}.npy"
            np.save(depth_path, depth_map.astype(np.float32))
            
            return str(depth_path)
            
        elif self.X_canon is not None:
            # Fallback: reconstruct depth map from sparse points
            output_dir = pathlib.Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create depth map
            img_shape_flat = self.img_shape.flatten()
            height, width = int(img_shape_flat[0].item()), int(img_shape_flat[1].item())
            depth_map = torch.zeros(height, width, dtype=torch.float32)
            
            if self.K is not None:
                # Use camera intrinsics for projection
                points_2d = self.K @ self.X_canon.T  # Shape: [3, N]
                points_2d = points_2d[:2] / (points_2d[2:3] + 1e-8)  # Avoid division by zero
                points_2d = points_2d.T.round().long()  # Shape: [N, 2]
                
                # Use Euclidean distance for consistency with dense method
                depths = torch.norm(self.X_canon, dim=1)  # Euclidean distance
                
                # Filter valid points (within image bounds and positive depth)
                valid = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height) & \
                        (depths > 1e-8)  # Small positive threshold
                
                # Fill depth map
                if valid.sum() > 0:
                    valid_points = points_2d[valid]
                    valid_depths = depths[valid]
                    
                    for i in range(len(valid_points)):
                        y, x = valid_points[i, 1], valid_points[i, 0]
                        current_depth = depth_map[y, x]
                        if current_depth == 0 or valid_depths[i] < current_depth:
                            depth_map[y, x] = valid_depths[i]
            else:
                # No camera intrinsics - create a simple dense depth map from sparse points
                # This is a simplified approach: distribute sparse depths across nearby pixels
                depths = torch.norm(self.X_canon, dim=1)  # Euclidean distance
                
                if len(depths) > 0 and depths.max() > 1e-8:
                    # Simple approach: fill a small region around each point
                    num_points = min(len(self.X_canon), height * width // 4)  # Limit points
                    
                    # Sample points evenly across the image
                    for i in range(num_points):
                        # Distribute points across image
                        y = int((i % int(np.sqrt(num_points))) * height / np.sqrt(num_points))
                        x = int((i // int(np.sqrt(num_points))) * width / np.sqrt(num_points))
                        
                        # Use depth from corresponding sparse point
                        if i < len(depths):
                            depth_val = depths[i % len(depths)]
                            if depth_val > 1e-8:
                                # Fill a small region around this point
                                for dy in range(-2, 3):
                                    for dx in range(-2, 3):
                                        py, px = y + dy, x + dx
                                        if 0 <= py < height and 0 <= px < width:
                                            if depth_map[py, px] == 0:
                                                depth_map[py, px] = depth_val
            
            # Save depth map with consistent frame_id naming
            depth_path = output_dir / f"{self.frame_id:06d}.npy"
            np.save(depth_path, depth_map.cpu().numpy())
            
            print(f"Sparse depth map for frame {self.frame_id}: {(depth_map > 0).sum().item()}/{depth_map.numel()} non-zero pixels")
            return str(depth_path)
        else:
            # No depth information available
            print(f"No depth information available for frame {self.frame_id}")
            return None


def create_frame(i, img, T_WC, img_size=512, device="cuda:0"):
    img = resize_img(img, img_size)
    rgb = img["img"].to(device=device)
    img_shape = torch.tensor(img["true_shape"], device=device)
    img_true_shape = img_shape.clone()
    uimg = torch.from_numpy(img["unnormalized_img"]) / 255.0
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        uimg = uimg[::downsample, ::downsample]
        img_shape = img_shape // downsample
    frame = Frame(i, rgb, img_shape, img_true_shape, uimg, T_WC)
    
    # Set intrinsics matrix K if loaded from file
    intrinsics_path = config.get("intrinsics_file")
    loaded_intrinsics = load_intrinsics_file(intrinsics_path)
    if loaded_intrinsics is not None:
        # Create camera intrinsics matrix from loaded values
        fx = loaded_intrinsics['fx']
        fy = loaded_intrinsics['fy']
        cx = loaded_intrinsics['ox']
        cy = loaded_intrinsics['oy']
        
        # Get original image dimensions (this should match the intrinsics)
        # We need to determine the original image size that matches the intrinsics
        # For now, we'll assume a typical high-resolution smartphone image
        # In practice, this should be derived from the actual input image dimensions
        # before any resizing, or stored in the intrinsics file
        
        # Since we can't easily get the original image size at this point,
        # we'll estimate based on the provided intrinsics principal point
        # This is a reasonable approximation for smartphone cameras
        orig_width = int(cx * 2)  # Assume principal point is roughly centered
        orig_height = int(cy * 2)
        
        # Get transformation parameters for the current image processing pipeline
        dummy_img = np.zeros((orig_height, orig_width, 3))
        _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
            dummy_img, img_size, return_transformation=True
        )
        
        # Apply the same transformation logic as the Intrinsics class
        K_orig = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        K_frame = K_orig.copy()
        K_frame[0, 0] = K_orig[0, 0] / scale_w  # fx scaling
        K_frame[1, 1] = K_orig[1, 1] / scale_h  # fy scaling  
        K_frame[0, 2] = K_orig[0, 2] / scale_w - half_crop_w  # cx translation
        K_frame[1, 2] = K_orig[1, 2] / scale_h - half_crop_h  # cy translation
        
        # Apply downsampling if needed
        if downsample > 1:
            K_frame[0, 0] /= downsample  # fx
            K_frame[1, 1] /= downsample  # fy
            K_frame[0, 2] /= downsample  # cx
            K_frame[1, 2] /= downsample  # cy
        
        # Create intrinsics matrix
        K = torch.tensor(K_frame, device=device, dtype=torch.float32)
        frame.K = K
        
        print(f"Set K matrix for frame {i} from intrinsics file:")
        print(f"  Original: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"  Scaled: fx={K_frame[0,0]:.1f}, fy={K_frame[1,1]:.1f}, cx={K_frame[0,2]:.1f}, cy={K_frame[1,2]:.1f}")
        print(f"  Transformation: scale_w={scale_w:.3f}, scale_h={scale_h:.3f}, crop_w={half_crop_w:.1f}, crop_h={half_crop_h:.1f}, downsample={downsample}")
    
    return frame


class SharedStates:
    def __init__(self, manager, h, w, dtype=torch.float32, device="cuda"):
        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device

        self.lock = manager.RLock()
        self.paused = manager.Value("i", 0)
        self.mode = manager.Value("i", Mode.INIT)
        self.reloc_sem = manager.Value("i", 0)
        self.global_optimizer_tasks = manager.list()
        self.edges_ii = manager.list()
        self.edges_jj = manager.list()

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # fmt:off
        # shared state for the current frame (used for reloc/visualization)
        self.dataset_idx = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(3, h, w, device=device, dtype=dtype).share_memory_()
        self.uimg = torch.zeros(h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = lietorch.Sim3.Identity(1, device=device, dtype=dtype).data.share_memory_()
        self.X = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(h * w, 1, device=device, dtype=dtype).share_memory_()
        self.feat = torch.zeros(1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        # fmt: on

    def set_frame(self, frame):
        with self.lock:
            self.dataset_idx[:] = frame.frame_id
            self.img[:] = frame.img
            self.uimg[:] = frame.uimg
            self.img_shape[:] = frame.img_shape
            self.img_true_shape[:] = frame.img_true_shape
            self.T_WC[:] = frame.T_WC.data
            self.X[:] = frame.X_canon
            self.C[:] = frame.C
            self.feat[:] = frame.feat
            self.pos[:] = frame.pos

    def get_frame(self):
        with self.lock:
            frame = Frame(
                int(self.dataset_idx[0]),
                self.img,
                self.img_shape,
                self.img_true_shape,
                self.uimg,
                lietorch.Sim3(self.T_WC),
            )
            frame.X_canon = self.X
            frame.C = self.C
            frame.feat = self.feat
            frame.pos = self.pos
            return frame

    def queue_global_optimization(self, idx):
        with self.lock:
            self.global_optimizer_tasks.append(idx)

    def queue_reloc(self):
        with self.lock:
            self.reloc_sem.value += 1

    def dequeue_reloc(self):
        with self.lock:
            if self.reloc_sem.value == 0:
                return
            self.reloc_sem.value -= 1

    def get_mode(self):
        with self.lock:
            return self.mode.value

    def set_mode(self, mode):
        with self.lock:
            self.mode.value = mode

    def pause(self):
        with self.lock:
            self.paused.value = 1

    def unpause(self):
        with self.lock:
            self.paused.value = 0

    def is_paused(self):
        with self.lock:
            return self.paused.value == 1


class SharedKeyframes:
    def __init__(self, manager, h, w, buffer=512, dtype=torch.float32, device="cuda"):
        self.lock = manager.RLock()
        self.n_size = manager.Value("i", 0)

        self.h, self.w = h, w
        self.buffer = buffer
        self.dtype = dtype
        self.device = device

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # fmt:off
        self.dataset_idx = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(buffer, 3, h, w, device=device, dtype=dtype).share_memory_()
        self.uimg = torch.zeros(buffer, h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, device=device, dtype=dtype).share_memory_()
        self.X = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(buffer, h * w, 1, device=device, dtype=dtype).share_memory_()
        self.N = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.N_updates = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.feat = torch.zeros(buffer, 1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(buffer, 1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        self.is_dirty = torch.zeros(buffer, 1, device=device, dtype=torch.bool).share_memory_()
        self.K = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()
        self.dense_depth = torch.zeros(buffer, h, w, 3, device=device, dtype=dtype).share_memory_()
        # fmt: on

    def __getitem__(self, idx) -> Frame:
        with self.lock:
            # put all of the data into a frame
            kf = Frame(
                int(self.dataset_idx[idx]),
                self.img[idx],
                self.img_shape[idx],
                self.img_true_shape[idx],
                self.uimg[idx],
                lietorch.Sim3(self.T_WC[idx]),
            )
            kf.X_canon = self.X[idx]
            kf.C = self.C[idx]
            kf.feat = self.feat[idx]
            kf.pos = self.pos[idx]
            kf.N = int(self.N[idx])
            kf.N_updates = int(self.N_updates[idx])
            kf.dense_depth = self.dense_depth[idx]
            if config["use_calib"]:
                kf.K = self.K
            return kf

    def __setitem__(self, idx, value: Frame) -> None:
        with self.lock:
            self.n_size.value = max(idx + 1, self.n_size.value)

            # set the attributes
            self.dataset_idx[idx] = value.frame_id
            self.img[idx] = value.img
            self.uimg[idx] = value.uimg
            self.img_shape[idx] = value.img_shape
            self.img_true_shape[idx] = value.img_true_shape
            self.T_WC[idx] = value.T_WC.data
            self.X[idx] = value.X_canon
            self.C[idx] = value.C
            self.feat[idx] = value.feat
            self.pos[idx] = value.pos
            self.N[idx] = value.N
            self.N_updates[idx] = value.N_updates
            if value.dense_depth is not None:
                self.dense_depth[idx] = value.dense_depth
            self.is_dirty[idx] = True
            return idx

    def __len__(self):
        with self.lock:
            return self.n_size.value

    def append(self, value: Frame):
        with self.lock:
            self[self.n_size.value] = value

    def pop_last(self):
        with self.lock:
            self.n_size.value -= 1

    def last_keyframe(self) -> Optional[Frame]:
        with self.lock:
            if self.n_size.value == 0:
                return None
            return self[self.n_size.value - 1]

    def update_T_WCs(self, T_WCs, idx) -> None:
        with self.lock:
            self.T_WC[idx] = T_WCs.data

    def get_dirty_idx(self):
        with self.lock:
            idx = torch.where(self.is_dirty)[0]
            self.is_dirty[:] = False
            return idx

    def set_intrinsics(self, K):
        assert config["use_calib"]
        with self.lock:
            self.K[:] = K

    def get_intrinsics(self):
        assert config["use_calib"]
        with self.lock:
            return self.K
