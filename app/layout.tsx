import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "YOLO ONNX Web Demo",
  description: "Minimal YOLO ONNX object detection demo in the browser",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
