import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';

import { WordCheck } from '../../../core/prompt/prompt-validation';
import { GeneratorStore } from '../../../core/state/generator.store';

/** Display order and labels of the vocabulary groups sent by the API. */
const GROUPS: { key: string; label: string }[] = [
  { key: 'sizes', label: 'Size' },
  { key: 'colors', label: 'Color' },
  { key: 'shapes', label: 'Shape' },
  { key: 'other', label: 'Position and linking words' },
];

/** Step 1: write the prompt, with live validation, word shortcuts and examples. */
@Component({
  selector: 'app-prompt-panel',
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './prompt-panel.component.html',
  styleUrl: './prompt-panel.component.scss',
})
export class PromptPanelComponent {
  protected readonly store = inject(GeneratorStore);
  protected readonly exampleGroup = signal(0);

  protected readonly words = computed(() => this.store.check()?.words ?? []);
  protected readonly maxWords = computed(() => this.store.vocabulary()?.max_words ?? 0);

  /** Unknown words, each listed once, with their suggested fixes. */
  protected readonly unknownWords = computed<WordCheck[]>(() => {
    const byText = new Map<string, WordCheck>();
    for (const word of this.words()) {
      if (!word.known && !byText.has(word.text)) byText.set(word.text, word);
    }
    return [...byText.values()];
  });

  protected readonly wordGroups = computed(() => {
    const groups = this.store.vocabulary()?.groups ?? {};
    return GROUPS.filter((g) => (groups[g.key]?.length ?? 0) > 0).map((g) => ({
      label: g.label,
      words: groups[g.key],
    }));
  });

  protected readonly examples = computed(() => this.store.vocabulary()?.examples ?? []);
  protected readonly currentExamples = computed(() => this.examples()[this.exampleGroup()] ?? null);

  protected onInput(event: Event): void {
    this.store.setPrompt((event.target as HTMLTextAreaElement).value);
  }
}
