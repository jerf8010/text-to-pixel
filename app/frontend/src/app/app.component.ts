import { ChangeDetectionStrategy, Component, OnInit, inject } from '@angular/core';

import { GeneratorStore } from './core/state/generator.store';
import { PromptPanelComponent } from './features/generator/prompt-panel/prompt-panel.component';
import { ResultsPanelComponent } from './features/generator/results-panel/results-panel.component';
import { SettingsPanelComponent } from './features/generator/settings-panel/settings-panel.component';

/** Application shell: header, the three generator panels and the primary action. */
@Component({
  selector: 'app-root',
  imports: [PromptPanelComponent, SettingsPanelComponent, ResultsPanelComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent implements OnInit {
  protected readonly store = inject(GeneratorStore);

  ngOnInit(): void {
    this.store.load();
  }
}
