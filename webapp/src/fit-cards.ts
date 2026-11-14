// The per-area cards and rows of the Fit Analyser, so every screen draws
// a chest/shoulder/neck reading the same way.
import { esc } from './dom';
import type { FitRow } from './fit-report';

const MARK = { good: '✓', watch: '!', off: '✕' } as const;

export function areaCard(row: FitRow): string {
  return '<div class="fx-area">' +
    '<div class="fx-area-h">' +
      '<span class="fx-mark is-' + row.tone + '">' + MARK[row.tone] + '</span>' +
      '<span class="fx-area-l">' + esc(row.label) + '</span>' +
    '</div>' +
    '<div class="fx-area-v is-' + row.tone + '">' + esc(row.word) + '</div>' +
    '<div class="fx-area-line">' + esc(row.line) + '</div>' +
    (row.scored ? '' : '<div class="fx-area-note">Measured, not scored</div>') +
  '</div>';
}

export function secondaryRow(row: FitRow): string {
  return '<div class="fx-row">' +
    '<div class="fx-row-l">' + esc(row.label) + '</div>' +
    '<div class="fx-row-v is-' + row.tone + '">' + esc(row.word) + '</div>' +
    '<div class="fx-row-line">' + esc(row.line) + '</div>' +
  '</div>';
}
