import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';

import { ApiError } from '../../../core/api/api.models';
import { GeneratorStore } from '../../../core/state/generator.store';
import { PixelCanvasComponent } from '../../../shared/pixel-canvas.component';
import { pixelsToBlob } from '../../../shared/pixel-render';

const DOWNLOAD_SCALE = 16; // a 16x16 image becomes a 256x256 PNG

/** Shows the generated images, one card per model, side by side. */
@Component({
  selector: 'app-results-panel',
  imports: [PixelCanvasComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './results-panel.component.html',
  styleUrl: './results-panel.component.scss',
})
export class ResultsPanelComponent {
  protected readonly store = inject(GeneratorStore);
  protected readonly palette = computed(() => this.store.vocabulary()?.palette ?? []);
  /** One placeholder per image that is being generated. */
  protected readonly skeletonSlots = computed(() =>
    Array.from({ length: this.store.nVariants() }, (_, index) => index),
  );
  protected readonly seedCopied = signal(false);
  /** Short message announced to screen readers after copy / download actions. */
  protected readonly notice = signal<string | null>(null);

  protected errorTitle(error: ApiError): string {
    switch (error.code) {
      case 'network':
        return 'Cannot reach the server';
      case 'unknown_words':
      case 'prompt_too_long':
      case 'empty_prompt':
        return 'Check your description';
      case 'internal_error':
        return 'The model failed to generate an image';
      default:
        return 'Something went wrong';
    }
  }

  protected async copySeed(seed: number): Promise<void> {
    try {
      await navigator.clipboard.writeText(String(seed));
      this.seedCopied.set(true);
      this.notice.set('Seed copied to the clipboard.');
      setTimeout(() => this.seedCopied.set(false), 1500);
    } catch {
      this.notice.set('Could not copy the seed. Select it and copy it manually.');
    }
  }

  protected useSeed(seed: number): void {
    this.store.seedText.set(String(seed));
    this.notice.set(`Seed ${seed} will be used for the next generation.`);
  }

  protected async download(modelId: string, seed: number, index: number, pixels: number[][]): Promise<void> {
    try {
      const blob = await pixelsToBlob(pixels, this.palette(), DOWNLOAD_SCALE);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `pixelgen-${modelId}-${seed}-${index + 1}.png`;
      link.click();
      URL.revokeObjectURL(url);
      this.notice.set('Image downloaded.');
    } catch {
      this.notice.set('Could not create the PNG.');
    }
  }
}
