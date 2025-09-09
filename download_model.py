# python
from pathlib import Path
import numpy as np
from PIL import Image
import openvino as ov
import kagglehub

def find_saved_model_dir(root: Path) -> Path:
    # Look for a TF2 SavedModel folder (contains `saved_model.pb`)
    hits = list(root.rglob("saved_model.pb"))
    if hits:
        return hits[0].parent
    return root  # fallback if the download already points to SavedModel

def load_image_rgb(path: Path, size=(256, 256)) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    x = np.asarray(img).astype(np.float32) / 255.0  # scale to [0,1]
    x = np.expand_dims(x, axis=0)  # NHWC with batch=1 -> (1,H,W,3)
    return x

def main():
    # 1) Get/download the model (cached by kagglehub)
    dl_path = Path(kagglehub.model_download("google/movenet/tensorFlow2/singlepose-thunder"))
    saved_model_dir = find_saved_model_dir(dl_path)

    # 2) Convert TF SavedModel -> OpenVINO IR
    ov_model = ov.convert_model(saved_model_dir.as_posix())  # detects TF frontend
    out_dir = Path("models"); out_dir.mkdir(parents=True, exist_ok=True)
    ir_path = out_dir / "movenet_thunder.xml"
    ov.save_model(ov_model, ir_path.as_posix())

    # 3) Load and compile
    core = ov.Core()
    compiled = core.compile_model(ov_model, "CPU")

    # 4) Prepare input and infer
    # Replace with your image path
    image_path = Path("sample.jpg")
    inp = load_image_rgb(image_path, size=(256, 256))

    # If the converted model expects a different dtype, cast accordingly
    et = compiled.input(0).get_element_type()
    dtype_map = {
        ov.Type.f32: np.float32, ov.Type.f16: np.float16,
        ov.Type.u8: np.uint8, ov.Type.i32: np.int32
    }
    inp = inp.astype(dtype_map.get(et, np.float32), copy=False)

    res = compiled([inp])
    output = res[compiled.output(0)]
    print("IR saved to:", ir_path.as_posix())
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()