/** Drawing helpers for palette-indexed images. */

/** Paint `pixels` (palette indices) on `ctx` as crisp squares of `scale` canvas pixels. */
export function drawPixels(
  ctx: CanvasRenderingContext2D,
  pixels: number[][],
  palette: number[][],
  scale: number,
): void {
  ctx.imageSmoothingEnabled = false;
  pixels.forEach((row, y) => {
    row.forEach((index, x) => {
      const [r, g, b] = palette[index] ?? [255, 0, 255]; // magenta flags an invalid index
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x * scale, y * scale, scale, scale);
    });
  });
}

/** Render `pixels` to a PNG blob, `scale` times larger than the image. */
export function pixelsToBlob(pixels: number[][], palette: number[][], scale: number): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = (pixels[0]?.length ?? 0) * scale;
  canvas.height = pixels.length * scale;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return Promise.reject(new Error('Canvas is not available in this browser.'));
  }
  drawPixels(ctx, pixels, palette, scale);
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Could not encode the image.'))), 'image/png'),
  );
}
