import 'dart:math' as math;

/// One-Euro Filter — the industry standard for AR landmark smoothing.
///
/// Used by Snapchat, MediaPipe, ARKit internally.
/// Adapts smoothing dynamically:
///   - Slow/still → heavy smoothing (eliminates jitter)
///   - Fast movement → light smoothing (low latency, instant tracking)
///
/// Reference: https://cristal.univ-lille.fr/~casiez/1euro/
class OneEuroFilter {
  final double minCutoff;  // Minimum cutoff frequency (Hz). Lower = more smoothing.
  final double beta;       // Speed coefficient. Higher = less lag during fast moves.
  final double dCutoff;    // Derivative cutoff frequency.

  double _xPrev = 0;
  double _dxPrev = 0;
  double _tPrev = -1;
  bool _initialized = false;

  OneEuroFilter({
    this.minCutoff = 1.5,  // Tuned for AR: low jitter at rest
    this.beta = 0.5,       // Tuned for AR: responsive during movement
    this.dCutoff = 1.0,
  });

  double _smoothingFactor(double te, double cutoff) {
    final r = 2 * math.pi * cutoff * te;
    return r / (r + 1);
  }

  double _exponentialSmoothing(double a, double x, double xPrev) {
    return a * x + (1 - a) * xPrev;
  }

  double filter(double x, double t) {
    if (!_initialized) {
      _xPrev = x;
      _dxPrev = 0;
      _tPrev = t;
      _initialized = true;
      return x;
    }

    final te = t - _tPrev; // Time elapsed
    if (te <= 0) return _xPrev;

    // Estimate derivative (speed)
    final aD = _smoothingFactor(te, dCutoff);
    final dx = (x - _xPrev) / te;
    final dxSmoothed = _exponentialSmoothing(aD, dx, _dxPrev);

    // Adaptive cutoff based on speed
    final cutoff = minCutoff + beta * dxSmoothed.abs();

    // Filter the signal
    final a = _smoothingFactor(te, cutoff);
    final xFiltered = _exponentialSmoothing(a, x, _xPrev);

    _xPrev = xFiltered;
    _dxPrev = dxSmoothed;
    _tPrev = t;

    return xFiltered;
  }

  void reset() {
    _initialized = false;
  }
}

/// Filters a 2D landmark (x, y) with separate 1-Euro filters per axis.
class LandmarkFilter {
  final OneEuroFilter _xFilter;
  final OneEuroFilter _yFilter;

  LandmarkFilter({
    double minCutoff = 1.5,
    double beta = 0.5,
  })  : _xFilter = OneEuroFilter(minCutoff: minCutoff, beta: beta),
        _yFilter = OneEuroFilter(minCutoff: minCutoff, beta: beta);

  (double x, double y) filter(double x, double y, double t) {
    return (_xFilter.filter(x, t), _yFilter.filter(y, t));
  }

  void reset() {
    _xFilter.reset();
    _yFilter.reset();
  }
}
