// The reading the live try-on readout publishes the moment it settles, so the
// Fit Analyser quotes the same number. Was
// `window.frReading` in the single-file demo.
import type { FitReading } from './fit-report';

export interface LiveReading { reading: FitReading; garment: string }

export const fitSession = {
  frReading: null as LiveReading | null,
};
