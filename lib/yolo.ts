import * as ort from "onnxruntime-web";

export type PreprocessMeta = {
  originalWidth: number;
  originalHeight: number;
  modelWidth: number;
  modelHeight: number;
  scaleX: number;
  scaleY: number;
};

export type Detection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classId: number;
  label: string;
};

export type Thresholds = {
  scoreThreshold: number;
  iouThreshold: number;
};

export async function loadModel(modelPath: string): Promise<ort.InferenceSession> {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  ort.env.wasm.numThreads = 1;
  return ort.InferenceSession.create(modelPath, {
    executionProviders: ["webgpu", "wasm"],
    graphOptimizationLevel: "all",
  });
}

export async function preprocess(
  imageBitmap: ImageBitmap,
  modelSize = 640,
): Promise<{ tensor: ort.Tensor; meta: PreprocessMeta }> {
  const canvas = document.createElement("canvas");
  canvas.width = modelSize;
  canvas.height = modelSize;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  if (!ctx) {
    throw new Error("Failed to get canvas context for preprocessing");
  }

  ctx.drawImage(imageBitmap, 0, 0, modelSize, modelSize);
  const { data } = ctx.getImageData(0, 0, modelSize, modelSize);

  const floatData = new Float32Array(1 * 3 * modelSize * modelSize);
  const area = modelSize * modelSize;

  for (let i = 0; i < area; i += 1) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;

    floatData[i] = r;
    floatData[area + i] = g;
    floatData[area * 2 + i] = b;
  }

  const tensor = new ort.Tensor("float32", floatData, [1, 3, modelSize, modelSize]);
  const meta: PreprocessMeta = {
    originalWidth: imageBitmap.width,
    originalHeight: imageBitmap.height,
    modelWidth: modelSize,
    modelHeight: modelSize,
    scaleX: imageBitmap.width / modelSize,
    scaleY: imageBitmap.height / modelSize,
  };

  return { tensor, meta };
}

export async function runInference(
  session: ort.InferenceSession,
  tensor: ort.Tensor,
): Promise<ort.Tensor> {
  const inputName = session.inputNames[0];
  const feeds: Record<string, ort.Tensor> = { [inputName]: tensor };
  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const output = results[outputName];

  if (!output) {
    throw new Error(`Output tensor not found: ${outputName}`);
  }

  return output;
}

function iou(a: Detection, b: Detection): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const inter = interW * interH;
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function nms(detections: Detection[], iouThreshold: number): Detection[] {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const selected: Detection[] = [];

  while (sorted.length > 0) {
    const current = sorted.shift();
    if (!current) continue;

    selected.push(current);

    for (let i = sorted.length - 1; i >= 0; i -= 1) {
      if (sorted[i].classId !== current.classId) continue;
      if (iou(current, sorted[i]) > iouThreshold) {
        sorted.splice(i, 1);
      }
    }
  }

  return selected;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function postprocess(
  output: ort.Tensor,
  meta: PreprocessMeta,
  thresholds: Thresholds,
  labels: string[],
): Detection[] {
  const dims = output.dims;
  if (dims.length !== 3) {
    throw new Error(`Unsupported output shape: [${dims.join(", ")}]`);
  }

  const data = output.data as Float32Array;
  let numPreds = 0;
  let features = 0;
  let axisMode: "features-first" | "preds-first";

  if (dims[1] <= 128 && dims[2] >= 1000) {
    axisMode = "features-first";
    features = dims[1];
    numPreds = dims[2];
  } else {
    axisMode = "preds-first";
    numPreds = dims[1];
    features = dims[2];
  }

  const detections: Detection[] = [];
  const hasObjectness =
    features === labels.length + 5 ||
    (features !== labels.length + 4 && features > labels.length + 4);
  const classStart = hasObjectness ? 5 : 4;
  const classCount = Math.max(0, features - classStart);

  for (let i = 0; i < numPreds; i += 1) {
    const at = (f: number) => {
      if (axisMode === "features-first") {
        return data[f * numPreds + i];
      }
      return data[i * features + f];
    };

    const cx = at(0);
    const cy = at(1);
    const w = at(2);
    const h = at(3);

    let bestClassScore = 0;
    let bestClassId = -1;

    for (let c = 0; c < classCount; c += 1) {
      const score = at(classStart + c);
      if (score > bestClassScore) {
        bestClassScore = score;
        bestClassId = c;
      }
    }

    const objectness = hasObjectness ? at(4) : 1;
    const score = objectness * bestClassScore;

    if (score < thresholds.scoreThreshold || bestClassId < 0) {
      continue;
    }

    const x1 = clamp((cx - w / 2) * meta.scaleX, 0, meta.originalWidth);
    const y1 = clamp((cy - h / 2) * meta.scaleY, 0, meta.originalHeight);
    const x2 = clamp((cx + w / 2) * meta.scaleX, 0, meta.originalWidth);
    const y2 = clamp((cy + h / 2) * meta.scaleY, 0, meta.originalHeight);

    detections.push({
      x1,
      y1,
      x2,
      y2,
      score,
      classId: bestClassId,
      label: labels[bestClassId] ?? `class_${bestClassId}`,
    });
  }

  return nms(detections, thresholds.iouThreshold);
}

export function drawDetections(
  canvas: HTMLCanvasElement,
  image: CanvasImageSource,
  detections: Detection[],
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to get canvas context for drawing");
  }

  const w = (image as ImageBitmap).width;
  const h = (image as ImageBitmap).height;

  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(image, 0, 0, w, h);

  ctx.lineWidth = 2;
  ctx.font = "16px sans-serif";

  detections.forEach((det) => {
    const color = "#00d4ff";
    ctx.strokeStyle = color;
    ctx.fillStyle = color;

    const boxW = det.x2 - det.x1;
    const boxH = det.y2 - det.y1;
    ctx.strokeRect(det.x1, det.y1, boxW, boxH);

    const text = `${det.label} ${(det.score * 100).toFixed(1)}%`;
    const textMetrics = ctx.measureText(text);
    const textW = textMetrics.width + 10;
    const textH = 20;
    const textX = det.x1;
    const textY = Math.max(0, det.y1 - textH);

    ctx.fillRect(textX, textY, textW, textH);
    ctx.fillStyle = "#000";
    ctx.fillText(text, textX + 5, textY + 15);
  });
}
