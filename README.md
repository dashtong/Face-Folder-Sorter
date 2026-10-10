# Face Folder Sorter

A standalone Python tool that recursively scans a photo folder, groups detected
faces, lets you correct the groups in a browser, and prefixes each image with
the reviewed person names.

Reviewed people are remembered in a local SQLite database at
`data/people.sqlite3`. On later scans, matching faces start with the saved name,
so the same person does not need to be renamed again. The database stores names
and numerical face embeddings, not copies of the photos or face crops.

The Python application lives in `main.py`; its review interface is separated
into `index.html`, `styles.css`, and `app.js`. InsightFace model files are under
`data/models`. It is not a Python library and does not need to be installed as
a package.

## Install

Python 3.10–3.12 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements.txt` uses GPU-enabled ONNX Runtime. The application automatically
falls back to its CPU provider when CUDA is unavailable. For a smaller CPU-only
installation, replace `onnxruntime-gpu` in `requirements.txt` with
`onnxruntime`.

The CUDA and cuDNN runtime DLLs are installed in the virtual environment and
preloaded automatically, including when the application starts from the Windows
right-click menu.

The included `buffalo_m` files are loaded from `data/models`. Override their
location with the `FACE_MODEL_ROOT` environment variable if needed.

## Run

```powershell
python main.py "C:\path\to\photos"
python main.py --files "C:\photos\one.jpg" "C:\photos\two.png"
```

The folder and every subfolder are scanned for JPG, JPEG, PNG, BMP, WebP, TIFF,
and TIF images. A local review page opens automatically. Keep the terminal open
until you finish.

## Windows right-click menu

Install the context-menu entry once:

```powershell
.venv\Scripts\python.exe main.py --install-context-menu
```

You can then launch the sorter in either of these ways:

- right-click a folder and select **Sort photos by face**;
- open a folder, right-click its empty background, and select
  **Sort photos by face**;
- select one or more image files, right-click, and select
  **Sort selected photos by face** to process only that selection.

The selected-image action supports up to 100 files in one launch. On Windows 11,
the command may appear under **Show more options**. Explorer invokes legacy
commands once per selected item, so a lightweight hidden helper batches those
invocations and starts one scanner window containing the complete selection.

The registration is for the current Windows user and does not
require administrator access.

If the project is moved to another location, run the installation command again
so Windows receives the new `main.py` path. To remove the entry:

```powershell
.venv\Scripts\python.exe main.py --uninstall-context-menu
```

In the review page you can:

- edit a person's name in the group heading;
- drag a face into another group;
- right-click a face and select its destination;
- choose the maximum names in a filename prefix, defaulting to five;
- move unwanted or unusable faces into the permanent red `DROP` group.

When you click **Confirm & rename**, every explicitly named face is learned by
the local database. Placeholder names such as `Person 001` and faces in `DROP`
are not saved. Editing the heading of a previously saved person updates that
person's database name. Moving an incorrectly matched face to another group
corrects that face reference in the database.

`DROP` is always shown at the bottom, even when empty. Faces in `DROP` never add
a name to the image filename.

Click **Confirm & rename** to apply the result. Unique person names are sorted
in bytewise/ASCII order, producing names such as:

```text
Alice_Bob_original.jpg
Alice_another-photo.png
```

Images with no retained person remain unchanged. Renaming uses a two-stage move
to avoid overwriting files when names collide.

### Embedded face metadata

Confirmation also writes every reviewed non-`DROP` name into the image itself.
This list is independent of **Max names in prefix**: the filename may contain at
most five names while the embedded metadata contains all of them.

- JPEG and WebP store names as EXIF `XPKeywords` entries prefixed with `Face:`.
- PNG stores a UTF-8 JSON array in the standard `FaceNames` iTXt field.
- BMP and TIFF are left unchanged because they do not offer the same reliable
  metadata update path without rewriting their image encoding.

JPEG/WebP compressed pixels and PNG image-data chunks are copied unchanged, so
embedding names does not reduce image quality. Existing EXIF keywords not owned
by Face Folder Sorter are preserved. A future search engine can read all
`Face:*` keywords or the PNG `FaceNames` value. The writer detects the actual
file signature, so a JPEG incorrectly named with a `.png` extension is still
handled safely as JPEG.

## Options

```powershell
python main.py "C:\photos" --similarity 0.52
python main.py "C:\photos" --min-face-pixels 32
python main.py "C:\photos" --detector-sizes 640
python main.py "C:\photos" --detection-threshold 0.30
python main.py "C:\photos" --database "D:\Face Data\people.sqlite3"
```

The default detector sizes are `640,1280`. The 640 pass handles large close-up
faces better, while 1280 helps find smaller background faces. The results from
both passes are merged.

The review server binds only to `127.0.0.1` and stops after confirmation.
