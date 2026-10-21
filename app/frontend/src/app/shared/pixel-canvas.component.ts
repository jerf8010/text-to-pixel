import { ChangeDetectionStrategy, Component, ElementRef, effect, input, viewChild } from '@angular/core';

import { drawPixels } from './pixel-render';

/** Renders one palette-indexed image on a canvas, with crisp (non-smoothed) pixels. */
@Component({
  selector: 'app-pixel-canvas',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<canvas #surface role="img" [attr.aria-label]="label()"></canvas>`,
  styles: `
    :host {
      display: block;
    }
    canvas {
      display: block;
      width: 100%;
      height: auto;
      aspect-ratio: 1;
      image-rendering: pixelated;
      background: #000;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
    }
  `,
})
export class PixelCanvasComponent {
  readonly pixels = input.required<number[][]>();
  readonly palette = input.required<number[][]>();
  /** Accessible description of the image. */
  readonly label = input('Generated image');
  /** Canvas pixels per image pixel; the element itself scales with its container. */
  readonly scale = input(16);

  private readonly surface = viewChild<ElementRef<HTMLCanvasElement>>('surface');

  constructor() {
    effect(() => {
      const canvas = this.surface()?.nativeElement;
      if (!canvas) return; // the view is not created yet; the effect re-runs once it is
      const pixels = this.pixels();
      const scale = this.scale();
      // Setting the size also clears the canvas, so do it right before drawing.
      canvas.width = (pixels[0]?.length ?? 0) * scale;
      canvas.height = pixels.length * scale;
      const ctx = canvas.getContext('2d');
      if (ctx) drawPixels(ctx, pixels, this.palette(), scale);
    });
  }
}
