"""Run the ONNX tooth landmark model with standard FDI outputs."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
FDI_ORDER = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
IMG_H, IMG_W = (512, 1024)

def prepare(image):
    h, w = image.shape
    scale = min(IMG_W / w, IMG_H / h)
    nw, nh = (int(w * scale), int(h * scale))
    left, top = ((IMG_W - nw) // 2, (IMG_H - nh) // 2)
    canvas = np.zeros((IMG_H, IMG_W), dtype=image.dtype)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return (np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), (scale, left, top))

def decode_logits(heat, presence, image_shape, transform, confidence=0.55, decoder='softargmax'):
    scale, left, top = transform
    conf = 1 / (1 + np.exp(-np.clip(presence[0], -80, 80)))
    if decoder == 'softargmax':
        logits = np.stack([cv2.resize(channel, (512, 256), interpolation=cv2.INTER_LINEAR) for channel in heat[0]])
        logits = logits - logits.max(axis=(1, 2), keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=(1, 2), keepdims=True)
        xcoords = (weights.sum(axis=1) * np.arange(512, dtype=np.float32)).sum(axis=1)
        ycoords = (weights.sum(axis=2) * np.arange(256, dtype=np.float32)).sum(axis=1)
        h, w = (256, 512)
    elif decoder == 'hardargmax':
        h, w = heat.shape[-2:]
        indices = heat[0].reshape(32, -1).argmax(axis=1)
        ycoords, xcoords = np.divmod(indices, w)
    else:
        raise ValueError(f'Unknown decoder: {decoder}')
    predictions = []
    for i, fdi in enumerate(FDI_ORDER):
        exist = bool(conf[i] >= confidence)
        p = dict(fdi=fdi, exist=exist, confidence=float(conf[i]), x=None, y=None, x_lb=None, y_lb=None)
        if exist:
            p['x_lb'] = float(xcoords[i] / max(1, w - 1) * (IMG_W - 1))
            p['y_lb'] = float(ycoords[i] / max(1, h - 1) * (IMG_H - 1))
            p['x'] = float(np.clip((p['x_lb'] - left) / scale, 0, image_shape[1] - 1))
            p['y'] = float(np.clip((p['y_lb'] - top) / scale, 0, image_shape[0] - 1))
        predictions.append(p)
    return predictions

def predict(session, image, confidence=0.55, decoder='softargmax'):
    tensor, transform = prepare(image)
    heat, presence = session.run(['heatmap_logits_fdi', 'presence_logits_fdi'], {'image': tensor})
    return decode_logits(heat, presence, image.shape, transform, confidence, decoder)

def create_session(model, device='auto'):
    available = ort.get_available_providers()
    if device == 'cuda' and 'CUDAExecutionProvider' not in available:
        raise RuntimeError('CUDA provider unavailable; install onnxruntime-gpu and its runtime dependencies, or use --device cpu')
    providers = ['CPUExecutionProvider']
    if device in ('auto', 'cuda') and 'CUDAExecutionProvider' in available:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    if device == 'cuda' and 'CUDAExecutionProvider' not in session.get_providers():
        raise RuntimeError('CUDA initialization failed; check GPU runtime dependencies or use --device cpu')
    expected = json.dumps(FDI_ORDER)
    if session.get_modelmeta().custom_metadata_map.get('fdi_order') != expected:
        raise ValueError('Model FDI channel-order metadata does not match this inference interface')
    return session

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().with_name('toothmap_fdi.onnx'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('predictions'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--conf', type=float, default=0.55)
    parser.add_argument('--decoder', choices=['softargmax', 'hardargmax'], default='softargmax', help='softargmax: bilinear logits upsampling to 256x512; hardargmax: native 32x64 peak')
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error('Invalid confidence')
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    paths = sorted((p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in extensions)) if args.input.is_dir() else [args.input]
    if not paths:
        raise FileNotFoundError('No input images')
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('Input images must have unique filename stems')
    session = create_session(args.model, args.device)
    print('Providers:', session.get_providers(), flush=True)
    for folder in ('images_fdi', 'json_fdi', 'details'):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Cannot read {path}')
        predictions = predict(session, image, args.conf, args.decoder)
        visual = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        shapes = []
        for p in predictions:
            if not p['exist']:
                continue
            x, y = (round(p['x']), round(p['y']))
            cv2.circle(visual, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(visual, str(p['fdi']), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            shapes.append(dict(label=f'tooth_{p['fdi']:02d}', points=[[p['x'], p['y']]], group_id=None, shape_type='point', flags=dict(fdi=p['fdi'], exist=True, confidence=round(p['confidence'], 6))))
        labelme = dict(version='5.4.1', flags={}, shapes=shapes, imagePath=path.name, imageData=None, imageHeight=image.shape[0], imageWidth=image.shape[1])
        target = args.output / 'images_fdi' / f'{path.stem}_fdi.png'
        if not cv2.imwrite(str(target), visual):
            raise OSError(f'Cannot write {target}')
        (args.output / 'json_fdi' / f'{path.stem}_fdi.json').write_text(json.dumps(labelme, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.output / 'details' / f'{path.stem}.json').write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{path.name}: {len(shapes)} teeth', flush=True)
if __name__ == '__main__':
    main()
