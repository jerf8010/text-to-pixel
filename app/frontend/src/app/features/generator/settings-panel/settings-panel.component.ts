import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { GeneratorStore } from '../../../core/state/generator.store';

/** Step 2: choose which models to run, plus the advanced sampling options. */
@Component({
  selector: 'app-settings-panel',
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './settings-panel.component.html',
  styleUrl: './settings-panel.component.scss',
})
export class SettingsPanelComponent {
  protected readonly store = inject(GeneratorStore);
  protected readonly variantOptions = [1, 2, 3, 4];

  protected isSelected(id: string): boolean {
    return this.store.selectedModelIds().includes(id);
  }

  protected formatParams(count: number): string {
    return count >= 1_000_000 ? `${(count / 1_000_000).toFixed(2)}M` : `${Math.round(count / 1000)}K`;
  }

  protected onVariants(event: Event): void {
    this.store.nVariants.set(Number((event.target as HTMLSelectElement).value));
  }

  protected onTemperature(event: Event): void {
    this.store.temperature.set(Number((event.target as HTMLInputElement).value));
  }

  protected onSeed(event: Event): void {
    this.store.seedText.set((event.target as HTMLInputElement).value);
  }
}
