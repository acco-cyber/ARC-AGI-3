from dataclasses import replace
import threading
import time
import unittest

from load_parity_probe import ProbeConfig, local_url, quantile, run_phase, shape_plan, build_payload


class ProbeTests(unittest.TestCase):
    def test_fixed_shape_budget_and_backend_limits(self):
        config = ProbeConfig().validate()
        plan = shape_plan(config)
        self.assertEqual(len(plan), 64)
        self.assertEqual(sum(p['max_output_tokens'] for p in plan), 4096)
        self.assertEqual(config.active_slots, 10)
        self.assertEqual(config.populations, (25, 110, 110, 25))
        for bad in [replace(config, active_slots=11), replace(config, max_output_tokens=4096),
                    replace(config, requests_per_phase=20), replace(config, phase_seconds=901)]:
            with self.assertRaises(ValueError):
                bad.validate()

    def test_refuses_external_host_or_credentials(self):
        for url in ['https://example.com', 'http://gateway:8001/v1',
                    'http://secret@127.0.0.1/v1', 'http://127.0.0.1/v1?token=bad']:
            with self.assertRaises(ValueError):
                local_url(url)
        self.assertEqual(local_url('http://127.0.0.1:8001/v1/'), 'http://127.0.0.1:8001/v1')

    def test_tokenized_shapes_and_working_set_identity(self):
        class FakeTokenizer:
            def encode(self, text, **kwargs):
                return [ord(c) for c in text]
            def decode(self, tokens):
                return ''.join(chr(c) for c in tokens)
            def apply_chat_template(self, messages, **kwargs):
                return [0] * (5 + sum(len(m['content']) for m in messages))
        case = {'case':0, 'target_input_tokens':2048, 'max_output_tokens':64, 'seed':1}
        first, n = build_payload(FakeTokenizer(), case, 0, 'flashnext', 25)
        again, n2 = build_payload(FakeTokenizer(), {**case, 'case':25}, 0, 'flashnext', 25)
        distinct, n3 = build_payload(FakeTokenizer(), {**case, 'case':25}, 0, 'flashnext', 110)
        self.assertEqual((n,n2,n3), (2048,2048,2048))
        self.assertEqual(first['messages'],again['messages'])
        self.assertNotEqual(first['messages'],distinct['messages'])

    def test_pending_population_preserves_actual_request_budget(self):
        config = replace(ProbeConfig(), phase_seconds=2, request_timeout_seconds=1)
        payloads = [({'case': p['case']}, p['target_input_tokens']) for p in shape_plan(config)]
        for population in (25, 110):
            lock = threading.Lock()
            active = peak = 0
            def post(payload, timeout):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(.002)
                with lock:
                    active -= 1
                return {'ok': True, 'completion_tokens': 64,
                        'prompt_tokens': config.input_tokens[payload['case'] % len(config.input_tokens)]}
            report = run_phase(population, payloads, config, post)
            self.assertTrue(report['valid_transport_comparison'])
            self.assertEqual(report['completed'], 64)
            self.assertEqual(report['completion_tokens'], 4096)
            self.assertLessEqual(report['peak_active_http'], 10)
            self.assertLessEqual(peak, 10)
            self.assertEqual(len({r['case'] for r in report['rows']}), 64)
            self.assertEqual(len({r['state'] for r in report['rows']}), min(population,64))

    def test_missing_or_mismatched_usage_invalidates_comparison(self):
        config = replace(ProbeConfig(), phase_seconds=2, request_timeout_seconds=1)
        payloads = [({'case': p['case']}, p['target_input_tokens']) for p in shape_plan(config)]
        for response in [{'ok': True}, {'ok': True, 'prompt_tokens': 1, 'completion_tokens': 64}]:
            report = run_phase(25, payloads, config, lambda *args: response)
            self.assertFalse(report['valid_transport_comparison'])
            self.assertEqual(report['completed'], 0)

    def test_deadline_prevents_new_requests_and_drains_pending_states(self):
        config = replace(ProbeConfig(), phase_seconds=.02, request_timeout_seconds=.01)
        payloads = [({'case': p['case']}, p['target_input_tokens']) for p in shape_plan(config)]
        def post(payload, timeout):
            time.sleep(.03)
            return {'ok': True, 'completion_tokens': 64}
        report = run_phase(110, payloads, config, post)
        self.assertFalse(report['valid_transport_comparison'])
        self.assertLessEqual(report['attempted'],10)
        self.assertLess(report['elapsed_seconds'], .5)

    def test_quantile_handles_singleton_and_empty(self):
        self.assertIsNone(quantile([], .95))
        self.assertEqual(quantile([5], .95), 5)
        self.assertEqual(quantile([0,10], .5), 5)


if __name__ == '__main__':
    unittest.main()
