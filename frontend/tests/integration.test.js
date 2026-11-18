const test = require('node:test');
const assert = require('node:assert');

const BASE_URL = 'http://localhost:8000';

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
  assert.ok(tripIds.includes('00000000'));
  assert.ok(tripIds.includes('00000007'));
  assert.ok(tripIds.includes('00000091'));
});

test('3. POST /api/v1/predict - Inference and metric generation', async () => {
  const res = await fetch(`${BASE_URL}/api/v1/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trip_id: '00000000' })
  });
  assert.strictEqual(res.status, 200);
  const data = await res.json();
  
  assert.strictEqual(data.trip_id, '00000000');
  assert.strictEqual(data.status, 'success');
  assert.ok(data.metrics.top1_accuracy > 0.8);
  assert.ok(data.metrics.macro_f1 > 0.85);
  assert.ok(data.metrics.route_continuity > 0.88);
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
    body: JSON.stringify({ trip_id: 'invalid_trip_999' })
  });
  assert.strictEqual(res.status, 404);
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
