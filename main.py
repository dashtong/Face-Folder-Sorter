from __future__ import annotations
import argparse
import json
import os
import re
import struct
import sys
import threading
import uuid
import webbrowser
import zlib
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
DROP_GROUP = "DROP"


@dataclass(slots=True)
class FaceObservation:
    bbox: np.ndarray
    score: float
    embedding: np.ndarray


@dataclass(slots=True)
class ScannedFace:
    id: str
    image_id: str
    bbox: tuple[int, int, int, int]
    score: float
    embedding: np.ndarray
    group: str = ""
    low_quality: bool = False


@dataclass(slots=True)
class ScannedImage:
    id: str
    path: Path
    relative_path: str
    width: int
    height: int
    error: str | None = None


@dataclass(slots=True)
class ScanResult:
    root: Path
    images: list[ScannedImage]
    faces: list[ScannedFace]


class InsightFaceBackend:
    """Multi-scale SCRFD detection and ArcFace embeddings."""

    def __init__(
        self,
        detector_sizes: tuple[int, ...],
        detection_threshold: float,
    ):
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "Dependencies are missing. Run: python -m pip install -r requirements.txt"
            ) from exc

        requested = tuple(
            item.strip()
            for item in os.getenv(
                "FACE_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider"
            ).split(",")
            if item.strip()
        )
        available = set(ort.get_available_providers())
        providers = [provider for provider in requested if provider in available]
        if not providers:
            raise RuntimeError(
                f"Requested ONNX providers are unavailable. Available: {sorted(available)}"
            )
        model_root = Path(
            os.getenv("FACE_MODEL_ROOT", str(PROJECT_DIR / "data" / "models"))
        )
        self._app = FaceAnalysis(
            name=os.getenv("FACE_MODEL_NAME", "buffalo_m"),
            root=str(model_root),
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self._app.prepare(
            ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
            det_size=(detector_sizes[0], detector_sizes[0]),
            det_thresh=detection_threshold,
        )
        self._detector_sizes = detector_sizes

    def analyze(self, image: np.ndarray) -> list[FaceObservation]:
        observations: list[FaceObservation] = []
        detector = self._app.det_model
        original_size = detector.input_size
        try:
            for size in self._detector_sizes:
                detector.input_size = (size, size)
                for face in self._app.get(image):
                    observations.append(
                        FaceObservation(
                            bbox=np.asarray(face.bbox, dtype=np.float32),
                            score=float(face.det_score),
                            embedding=normalize(face.embedding),
                        )
                    )
        finally:
            detector.input_size = original_size
        return merge_face_observations(observations)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize an empty embedding")
    return vector / norm


def merge_face_observations(
    observations: list[FaceObservation], iou_threshold: float = 0.5
) -> list[FaceObservation]:
    kept: list[FaceObservation] = []
    for observation in sorted(observations, key=lambda item: item.score, reverse=True):
        if all(bbox_iou(observation.bbox, item.bbox) < iou_threshold for item in kept):
            kept.append(observation)
    return kept


def bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = max(float(left[0]), float(right[0])), max(float(left[1]), float(right[1]))
    x2, y2 = min(float(left[2]), float(right[2])), min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def scan_folder(
    root: Path,
    backend: InsightFaceBackend,
    similarity_threshold: float,
    min_face_pixels: int,
    progress=None,
) -> ScanResult:
    root = root.expanduser().resolve()
    paths = image_paths(root)
    images: list[ScannedImage] = []
    faces: list[ScannedFace] = []
    for position, path in enumerate(paths, 1):
        image_id = f"i{position}"
        try:
            image = read_image(path)
            height, width = image.shape[:2]
            images.append(
                ScannedImage(
                    image_id,
                    path.resolve(),
                    path.resolve().relative_to(root).as_posix(),
                    width,
                    height,
                )
            )
            for face_number, observation in enumerate(backend.analyze(image), 1):
                x1, y1, x2, y2 = clamp_bbox(observation.bbox, width, height)
                faces.append(
                    ScannedFace(
                        id=f"{image_id}f{face_number}",
                        image_id=image_id,
                        bbox=(x1, y1, x2, y2),
                        score=observation.score,
                        embedding=observation.embedding,
                        low_quality=min(x2 - x1, y2 - y1) < min_face_pixels,
                    )
                )
        except Exception as exc:
            images.append(
                ScannedImage(
                    image_id,
                    path.resolve(),
                    path.resolve().relative_to(root).as_posix(),
                    0,
                    0,
                    str(exc),
                )
            )
        if progress:
            progress(position, len(paths), path)
    cluster_faces(faces, similarity_threshold)
    return ScanResult(root, images, faces)


def cluster_faces(faces: list[ScannedFace], similarity_threshold: float) -> None:
    usable = sorted(
        (face for face in faces if not face.low_quality),
        key=lambda face: face.score,
        reverse=True,
    )
    clusters: list[list[ScannedFace]] = []
    centroids: list[np.ndarray] = []
    for face in usable:
        if centroids:
            similarities = np.asarray([float(np.dot(face.embedding, c)) for c in centroids])
            best = int(np.argmax(similarities))
        else:
            similarities, best = np.empty(0), 0
        if centroids and float(similarities[best]) >= similarity_threshold:
            clusters[best].append(face)
            centroids[best] = normalize(
                np.mean([member.embedding for member in clusters[best]], axis=0)
            )
        else:
            clusters.append([face])
            centroids.append(face.embedding.copy())
    clusters.sort(key=lambda group: (-len(group), min(face.id for face in group)))
    for number, cluster in enumerate(clusters, 1):
        for face in cluster:
            face.group = f"Person {number:03d}"
    for face in faces:
        if face.low_quality:
            face.group = DROP_GROUP


def sanitize_person_name(name: str) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub("-", name.strip())
    return re.sub(r"\s+", " ", cleaned).strip(" ._")[:80]


def build_rename_plan(
    scan: ScanResult, assignments: dict[str, str], max_names: int
) -> list[tuple[Path, Path]]:
    if max_names < 0:
        raise ValueError("max_names cannot be negative")
    faces_by_image: dict[str, list[ScannedFace]] = {}
    for face in scan.faces:
        faces_by_image.setdefault(face.image_id, []).append(face)
    plan: list[tuple[Path, Path]] = []
    reserved = {image.path.resolve() for image in scan.images}
    destinations: set[Path] = set()
    for image in scan.images:
        names = {
            sanitize_person_name(assignments.get(face.id, face.group))
            for face in faces_by_image.get(image.id, [])
        }
        names.difference_update({"", DROP_GROUP, "Too far", "Ungroupable"})
        if not names or max_names == 0:
            continue
        ordered = sorted(names, key=lambda value: value.encode("utf-8"))[:max_names]
        prefix = "_".join(ordered) + "_"
        destination = image.path.with_name(prefix + strip_known_prefixes(image.path.name, names))
        destination = unique_destination(
            destination, reserved, destinations, image.path.resolve()
        )
        if destination != image.path:
            plan.append((image.path, destination))
            destinations.add(destination.resolve())
    return plan


def strip_known_prefixes(filename: str, names: set[str]) -> str:
    candidates = sorted(names, key=len, reverse=True)
    while True:
        match = next((name for name in candidates if filename.startswith(name + "_")), None)
        if match is None:
            return filename
        filename = filename[len(match) + 1 :]


def unique_destination(
    desired: Path, reserved: set[Path], destinations: set[Path], source: Path
) -> Path:
    candidate, number = desired, 2
    while candidate.resolve() in destinations or (
        candidate.resolve() in reserved and candidate.resolve() != source
    ) or (candidate.exists() and candidate.resolve() != source):
        candidate = desired.with_name(f"{desired.stem} ({number}){desired.suffix}")
        number += 1
    return candidate


def apply_rename_plan(plan: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    staged: list[tuple[Path, Path, Path]] = []
    try:
        for source, destination in plan:
            temporary = source.with_name(f".{source.name}.face-sort-{uuid.uuid4().hex}.tmp")
            source.rename(temporary)
            staged.append((source, temporary, destination))
        completed: list[tuple[Path, Path]] = []
        for source, temporary, destination in staged:
            temporary.rename(destination)
            completed.append((source, destination))
        return completed
    except Exception:
        for source, temporary, destination in reversed(staged):
            if temporary.exists() and not source.exists():
                temporary.rename(source)
            elif destination.exists() and not source.exists():
                destination.rename(source)
        raise


def write_face_metadata(path: Path, names: list[str]) -> bool:
    with path.open("rb") as handle:
        signature = handle.read(12)
    # Use content signatures rather than extensions.
    if signature.startswith(b"\xff\xd8\xff") or (
        signature.startswith(b"RIFF") and signature[8:12] == b"WEBP"
    ):
        write_exif_face_keywords(path, names)
        return True
    if signature.startswith(b"\x89PNG\r\n\x1a\n"):
        write_png_face_names(path, names)
        return True
    return False


def write_exif_face_keywords(path: Path, names: list[str]) -> None:
    import piexif

    try:
        exif = piexif.load(str(path))
    except ValueError:
        # WebP files commonly have no EXIF container yet.
        exif = {
            "0th": {},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
            "thumbnail": None,
        }
    tag = piexif.ImageIFD.XPKeywords
    existing_value = exif["0th"].get(tag, b"")
    if isinstance(existing_value, tuple):
        existing_value = bytes(existing_value)
    try:
        existing = existing_value.decode("utf-16le").rstrip("\x00").split(";")
    except (AttributeError, UnicodeDecodeError):
        existing = []
    # Face: distinguishes sorter-owned entries from the user's other keywords.
    keywords = [value for value in existing if value and not value.startswith("Face:")]
    keywords.extend(f"Face:{name}" for name in names)
    if keywords:
        exif["0th"][tag] = (";".join(dict.fromkeys(keywords)) + "\x00").encode(
            "utf-16le"
        )
    else:
        exif["0th"].pop(tag, None)
    piexif.insert(piexif.dump(exif), str(path))


def write_png_face_names(path: Path, names: list[str]) -> None:
    content = path.read_bytes()
    signature = b"\x89PNG\r\n\x1a\n"
    if not content.startswith(signature):
        raise ValueError(f"Invalid PNG file: {path}")
    keyword_prefix = b"FaceNames\x00"
    payload = keyword_prefix + b"\x00\x00\x00\x00" + json.dumps(
        names, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    new_chunk = make_png_chunk(b"iTXt", payload)
    output = bytearray(signature)
    offset = len(signature)
    inserted = False
    while offset < len(content):
        if offset + 12 > len(content):
            raise ValueError(f"Damaged PNG structure: {path}")
        length = struct.unpack(">I", content[offset : offset + 4])[0]
        end = offset + 12 + length
        chunk = content[offset:end]
        chunk_type = content[offset + 4 : offset + 8]
        chunk_data = content[offset + 8 : offset + 8 + length]
        if chunk_type == b"iTXt" and chunk_data.startswith(keyword_prefix):
            offset = end
            continue
        if chunk_type == b"IEND" and not inserted:
            output.extend(new_chunk)
            inserted = True
        output.extend(chunk)
        offset = end
    if not inserted:
        raise ValueError(f"PNG has no IEND chunk: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.metadata.tmp")
    try:
        temporary.write_bytes(output)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def make_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def face_crop_jpeg(scan: ScanResult, face: ScannedFace, size: int = 220) -> bytes:
    image = read_image(next(item.path for item in scan.images if item.id == face.image_id))
    x1, y1, x2, y2 = face.bbox
    pad_x, pad_y = int((x2 - x1) * 0.35), int((y2 - y1) * 0.35)
    crop = image[
        max(0, y1 - pad_y) : min(image.shape[0], y2 + pad_y),
        max(0, x1 - pad_x) : min(image.shape[1], x2 + pad_x),
    ]
    if crop.size == 0:
        raise ValueError("Empty face crop")
    height, width = crop.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(crop, (max(1, round(width * scale)), max(1, round(height * scale))))
    canvas = np.full((size, size, 3), 24, dtype=np.uint8)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("Could not encode crop")
    return encoded.tobytes()


def clamp_bbox(bbox: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox[:4])
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group every face in an image folder and review names in a browser."
    )
    parser.add_argument("folder", type=Path, nargs="?", help="Folder scanned recursively")
    context_menu = parser.add_mutually_exclusive_group()
    context_menu.add_argument(
        "--install-context-menu",
        action="store_true",
        help="Add Face Folder Sorter to the Windows folder context menu",
    )
    context_menu.add_argument(
        "--uninstall-context-menu",
        action="store_true",
        help="Remove Face Folder Sorter from the Windows folder context menu",
    )
    parser.add_argument("--port", type=int, default=0, help="Local port (default: automatic)")
    parser.add_argument("--similarity", type=float, default=0.48, help="Face clustering similarity")
    parser.add_argument("--min-face-pixels", type=int, default=40, help="Place smaller faces in DROP")
    parser.add_argument(
        "--detector-sizes",
        default="640,1280",
        help="Comma-separated detection resolutions (default: 640,1280)",
    )
    parser.add_argument("--detection-threshold", type=float, default=0.35, help="Minimum detection confidence (default: 0.35)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.install_context_menu:
        install_context_menu()
        print("Installed 'Sort photos by face' in the Windows folder context menu.")
        return 0
    if args.uninstall_context_menu:
        uninstall_context_menu()
        print("Removed Face Folder Sorter from the Windows folder context menu.")
        return 0
    if args.folder is None:
        raise SystemExit("error: provide a folder or use --install-context-menu")
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"error: not a folder: {folder}")
    print(f"Scanning images recursively in {folder}")
    try:
        detector_sizes = tuple(
            dict.fromkeys(int(value.strip()) for value in args.detector_sizes.split(","))
        )
        if not detector_sizes or any(size < 160 for size in detector_sizes):
            raise ValueError
    except ValueError:
        raise SystemExit("error: --detector-sizes must contain integers of at least 160")
    backend = InsightFaceBackend(
        detector_sizes=detector_sizes,
        detection_threshold=args.detection_threshold,
    )
    scan = scan_folder(
        folder,
        backend,
        similarity_threshold=args.similarity,
        min_face_pixels=args.min_face_pixels,
        progress=lambda done, total, path: print(f"[{done}/{total}] {path.name}", flush=True),
    )
    if not scan.images:
        raise SystemExit("error: no supported images found")
    app = ReviewApplication(scan)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Detected {len(scan.faces)} face(s) in {len(scan.images)} image(s).")
    print(f"Review the results at {url}")
    print("Keep this terminal open until you click Confirm & rename.")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped without further changes.")
    finally:
        server.server_close()
    return 0


def install_context_menu() -> None:
    if os.name != "nt":
        raise RuntimeError("The context-menu integration is available only on Windows")
    import winreg

    python = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = Path(sys.executable).resolve()
    script = Path(__file__).resolve()
    entries = (
        (r"Software\Classes\Directory\shell\FaceFolderSorter", "%1"),
        (r"Software\Classes\Directory\Background\shell\FaceFolderSorter", "%V"),
    )
    for key_path, folder_placeholder in entries:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "Sort photos by face")
            winreg.SetValueEx(key, "Icon", 0, winreg.REG_SZ, str(python))
        command = (
            f'cmd.exe /d /s /c ""{python}" "{script}" '
            f'"{folder_placeholder}" || pause"'
        )
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path + r"\command") as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, command)


def uninstall_context_menu() -> None:
    if os.name != "nt":
        raise RuntimeError("The context-menu integration is available only on Windows")
    import winreg

    entries = (
        r"Software\Classes\Directory\shell\FaceFolderSorter",
        r"Software\Classes\Directory\Background\shell\FaceFolderSorter",
    )
    for key_path in entries:
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path + r"\command")
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)
        except FileNotFoundError:
            pass


class ReviewApplication:
    def __init__(self, scan: ScanResult):
        self.scan = scan
        self.faces = {face.id: face for face in scan.faces}
        self.crop_cache: dict[str, bytes] = {}
        self.confirmed = False
        self.lock = threading.Lock()

    def data(self) -> dict:
        image_lookup = {image.id: image for image in self.scan.images}
        return {
            "root": str(self.scan.root),
            "images": len(self.scan.images),
            "errors": sum(image.error is not None for image in self.scan.images),
            "faces": [
                {
                    "id": face.id,
                    "imageId": face.image_id,
                    "file": image_lookup[face.image_id].relative_path,
                    "group": face.group,
                    "score": round(face.score, 3),
                    "lowQuality": face.low_quality,
                    "crop": f"/crop?id={face.id}",
                }
                for face in self.scan.faces
            ],
        }

    def confirm(self, payload: dict) -> dict:
        with self.lock:
            if self.confirmed:
                return {"ok": False, "error": "This scan was already confirmed."}
            assignments = payload.get("assignments")
            if not isinstance(assignments, dict) or set(assignments) != set(self.faces):
                return {"ok": False, "error": "The face assignment list is incomplete."}
            if not all(isinstance(value, str) for value in assignments.values()):
                return {"ok": False, "error": "Every face name must be text."}
            try:
                max_names = int(payload.get("maxNames", 5))
                if not 0 <= max_names <= 100:
                    raise ValueError
            except (TypeError, ValueError):
                return {"ok": False, "error": "Maximum names must be from 0 to 100."}
            try:
                plan = build_rename_plan(self.scan, assignments, max_names)
                renamed = apply_rename_plan(plan)
            except Exception as exc:
                return {"ok": False, "error": f"Rename failed: {exc}"}
            renamed_paths = {source.resolve(): destination for source, destination in renamed}
            faces_by_image: dict[str, list[ScannedFace]] = {}
            for face in self.scan.faces:
                faces_by_image.setdefault(face.image_id, []).append(face)
            metadata_written = 0
            metadata_skipped: list[str] = []
            metadata_errors: list[str] = []
            for image in self.scan.images:
                image_faces = faces_by_image.get(image.id, [])
                if not image_faces:
                    continue
                names = sorted(
                    {
                        sanitize_person_name(assignments.get(face.id, face.group))
                        for face in image_faces
                    }.difference({"", DROP_GROUP, "Too far", "Ungroupable"}),
                    key=lambda value: value.encode("utf-8"),
                )
                current_path = renamed_paths.get(image.path.resolve(), image.path)
                try:
                    if write_face_metadata(current_path, names):
                        metadata_written += 1
                    else:
                        metadata_skipped.append(current_path.name)
                except Exception as exc:
                    metadata_errors.append(f"{current_path.name}: {exc}")
            self.confirmed = True
            return {
                "ok": True,
                "renamed": len(renamed),
                "unchanged": len(self.scan.images) - len(renamed),
                "metadataWritten": metadata_written,
                "metadataSkipped": metadata_skipped,
                "metadataErrors": metadata_errors,
            }


def make_handler(app: ReviewApplication):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            static_files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            }
            if parsed.path in static_files:
                filename, content_type = static_files[parsed.path]
                try:
                    self._send(200, (PROJECT_DIR / filename).read_bytes(), content_type)
                except OSError as exc:
                    self._json(500, {"error": f"Could not load {filename}: {exc}"})
            elif parsed.path == "/api/data":
                self._json(200, app.data())
            elif parsed.path == "/crop":
                face_id = parse_qs(parsed.query).get("id", [""])[0]
                face = app.faces.get(face_id)
                if face is None:
                    self._json(404, {"error": "Unknown face"})
                    return
                try:
                    if face_id not in app.crop_cache:
                        app.crop_cache[face_id] = face_crop_jpeg(app.scan, face)
                    self._send(200, app.crop_cache[face_id], "image/jpeg")
                except Exception as exc:
                    self._json(500, {"error": str(exc)})
            else:
                self._json(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/confirm":
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 5_000_000:
                    raise ValueError("Request is too large")
                payload = json.loads(self.rfile.read(length))
                result = app.confirm(payload)
                self._json(200 if result["ok"] else 400, result)
                if result["ok"]:
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
            except Exception as exc:
                self._json(400, {"ok": False, "error": str(exc)})

        def _json(self, status: int, value: dict):
            self._send(status, json.dumps(value).encode(), "application/json")

        def _send(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


if __name__ == "__main__":
    raise SystemExit(main())
