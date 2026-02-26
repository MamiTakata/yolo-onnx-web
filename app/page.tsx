"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import {
  Detection,
  drawDetections,
  loadModel,
  postprocess,
  preprocess,
  runInference,
} from "@/lib/yolo";

type ModelStatus = "idle" | "loading" | "loaded" | "error";

const MODEL_PATH = "/models/yolov8n.onnx";
const LABELS_PATH = "/models/coco_labels.json";

export default function Home() {
  const [modelStatus, setModelStatus] = useState<ModelStatus>("idle");
  const [error, setError] = useState<string>("");
  const [scoreThreshold, setScoreThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [detections, setDetections] = useState<Detection[]>([]);

  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const imageRef = useRef<ImageBitmap | null>(null);
  const labelsRef = useRef<string[]>([]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    void fetch(LABELS_PATH)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to load labels: ${res.status}`);
        }
        return res.json() as Promise<string[]>;
      })
      .then((labels) => {
        labelsRef.current = labels;
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : "Unknown labels loading error";
        setError(msg);
      });
  }, []);

  async function ensureModelLoaded() {
    if (sessionRef.current) {
      return sessionRef.current;
    }

    setModelStatus("loading");
    setError("");

    try {
      const session = await loadModel(MODEL_PATH);
      sessionRef.current = session;
      setModelStatus("loaded");
      return session;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown model loading error";
      setError(msg);
      setModelStatus("error");
      throw e;
    }
  }

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    setError("");
    setDetections([]);
    setInferenceMs(null);

    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);

    try {
      const bitmap = await createImageBitmap(file);
      imageRef.current = bitmap;

      const canvas = canvasRef.current;
      if (!canvas) return;

      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        throw new Error("Failed to get canvas context");
      }

      ctx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to load image";
      setError(msg);
    }
  }

  async function handleDetect() {
    setError("");

    if (!imageRef.current) {
      setError("Please upload an image first");
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) {
      setError("Canvas is not ready");
      return;
    }

    try {
      const session = await ensureModelLoaded();
      const start = performance.now();

      const { tensor, meta } = await preprocess(imageRef.current, 640);
      const output = await runInference(session, tensor);
      const parsed = postprocess(
        output,
        meta,
        { scoreThreshold, iouThreshold },
        labelsRef.current,
      );

      drawDetections(canvas, imageRef.current, parsed);

      setDetections(parsed);
      setInferenceMs(performance.now() - start);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Inference failed";
      setError(msg);
    }
  }

  return (
    <main className="container">
      <h1>YOLO ONNX Browser Demo</h1>

      <section className="panel">
        <p>
          Model: <strong>{MODEL_PATH}</strong>
        </p>
        <p>
          Status:{" "}
          <strong>
            {modelStatus === "idle" && "Idle"}
            {modelStatus === "loading" && "Loading..."}
            {modelStatus === "loaded" && "Loaded"}
            {modelStatus === "error" && "Error"}
          </strong>
        </p>

        <div className="controls">
          <label htmlFor="file">Upload Image</label>
          <input id="file" type="file" accept="image/*" onChange={handleFileChange} />
          {fileName ? <small>{fileName}</small> : null}
        </div>

        <div className="controls">
          <label htmlFor="scoreThreshold">scoreThreshold: {scoreThreshold.toFixed(2)}</label>
          <input
            id="scoreThreshold"
            type="range"
            min="0.05"
            max="0.95"
            step="0.01"
            value={scoreThreshold}
            onChange={(e) => setScoreThreshold(Number(e.target.value))}
          />
        </div>

        <div className="controls">
          <label htmlFor="iouThreshold">iouThreshold: {iouThreshold.toFixed(2)}</label>
          <input
            id="iouThreshold"
            type="range"
            min="0.05"
            max="0.95"
            step="0.01"
            value={iouThreshold}
            onChange={(e) => setIouThreshold(Number(e.target.value))}
          />
        </div>

        <button type="button" onClick={handleDetect}>
          Detect
        </button>

        {inferenceMs !== null ? (
          <p>
            Inference time: <strong>{inferenceMs.toFixed(1)} ms</strong>
          </p>
        ) : null}

        <p>
          Detections: <strong>{detections.length}</strong>
        </p>

        {error ? <p className="error">Error: {error}</p> : null}
      </section>

      <section className="panel">
        <canvas ref={canvasRef} className="resultCanvas" />
      </section>
    </main>
  );
}
