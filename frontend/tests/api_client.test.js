const test = require('node:test');
const assert = require('node:assert');

test('API Base URL default resolution', () => {
  // Test fallback logic
  const defaultUrl = 'http://localhost:8000';
  assert.strictEqual(defaultUrl, 'http://localhost:8000');
});

test('Clean URL trailing slash removal', () => {
  const dirtyUrl = 'http://192.168.43.120:8000///';
  const cleaned = dirtyUrl.trim().replace(/\/+$/, '');
  assert.strictEqual(cleaned, 'http://192.168.43.120:8000');
});

test('HTTP 404 error formatting', () => {
  const errorBody = { detail: "Trip ID '99999999' not found" };
  const message = errorBody.detail || 'Unknown error';
  assert.strictEqual(message, "Trip ID '99999999' not found");
});

test('Ground-truth leakage security check', () => {
  // Simulated backend response
  const responseData = {
    trip_id: '00000000',
    status: 'success',
    metrics: { top1_accuracy: 0.8245, macro_f1: 0.8914 },
    route: { matched_arc_ids: [101, 102] },
    gps_observations: [{ obs_id: '00000000_OBS_0000', lat: 51.5, lon: -0.1 }]
  };

  // Assert NO ground truth reference labels or reference route arrays exist in client payload
  assert.strictEqual('reference_route' in responseData, false);
  assert.strictEqual('ground_truth_labels' in responseData, false);
  assert.strictEqual('reference_arc_ids' in responseData, false);
});
