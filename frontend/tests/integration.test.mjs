import test from 'node:test';
import assert from 'node:assert/strict';

const BASE_URL = process.env.PATHSNAP_TEST_API_URL || 'http://localhost:8000';

test('1. GET /health - Server health and model loading', async () => {
  const res = await fetch(`${BASE_URL}/health`);
  assert.strictEqual(res.status, 200);
  const data = await res.json();
  assert.strictEqual(data.status, 'healthy');
  assert.strictEqual(data.model_algorithm, 'XGBoost');
  assert.strictEqual(data.model_loaded, true);
  assert.strictEqual(data.hmm_ready, true);
  assert.strictEqual(data.evaluator_trips_available, 10);
});

test('2. GET /api/v1/trips - All 10 evaluator trips returned', async () => {
  const res = await fetch(`${BASE_URL}/api/v1/trips`);
  assert.strictEqual(res.status, 200);
  const data = await res.json();
  assert.strictEqual(data.total, 10);
  assert.strictEqual(data.trips.length, 10);
  
  const tripIds = data.trips.map(t => t.trip_id);
  assert.ok(tripIds.includes('00000015'));
  assert.ok(tripIds.includes('00000023'));
  assert.ok(tripIds.includes('00000094'));
});

test('3. POST /api/v1/predict - Inference and metric generation', async () => {
  const res = await fetch(`${BASE_URL}/api/v1/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trip_id: '00000023' })
  });
  assert.strictEqual(res.status, 200);
  const data = await res.json();
  
  assert.strictEqual(data.trip_id, '00000023');
  assert.strictEqual(data.status, 'success');
  assert.strictEqual(data.metrics.top1_accuracy, null);
  assert.strictEqual(data.metrics.macro_f1, null);
  for (const key of ['route_arc_precision', 'route_arc_recall', 'route_arc_f1']) {
    assert.ok(Number.isFinite(data.metrics[key]) && data.metrics[key] >= 0 && data.metrics[key] <= 1);
  }
  const { route_arc_precision: precision, route_arc_recall: recall, route_arc_f1: f1 } = data.metrics;
  assert.ok(Math.abs(f1 - (precision + recall ? 2 * precision * recall / (precision + recall) : 0)) < 1e-9);
  assert.ok(data.metrics.route_continuity >= 0 && data.metrics.route_continuity <= 1);
  assert.ok(data.route.predicted_polyline.length > 0);
  assert.ok(data.gps_observations.length > 0);

  // Security constraint: Ground-truth reference MUST NOT be in response
  assert.strictEqual('reference_route' in data, false);
  assert.strictEqual('reference_labels' in data, false);
});

test('4. POST /api/v1/predict - 404 for invalid trip', async () => {
  const res = await fetch(`${BASE_URL}/api/v1/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trip_id: '99999999' })
  });
  assert.strictEqual(res.status, 404);
});

test('POST /api/v1/predict - 422 for malformed trip ID', async () => {
  const res = await fetch(`${BASE_URL}/api/v1/predict`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trip_id: '../invalid' })
  });
  assert.strictEqual(res.status, 422);
});

test('5. GET /api/v1/model/info - XGBoost metadata verification', async () => {
  const res = await fetch(`${BASE_URL}/api/v1/model/info`);
  assert.strictEqual(res.status, 200);
  const data = await res.json();
  assert.strictEqual(data.algorithm, 'XGBoost');
  assert.strictEqual(data.feature_count, 4);
  assert.deepStrictEqual(data.feature_names, [
    'distance_to_segment_m',
    'projection_fraction',
    'observation_speed_mps',
    'heading_difference_rad'
  ]);
});
