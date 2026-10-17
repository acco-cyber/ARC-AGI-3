"""CPU diagnostics must never silently acquire a GPU or accelerator fallback."""
import json
from pathlib import Path
import tempfile
import unittest

from kaggle_rest import push_payload


class ResourceGuardTests(unittest.TestCase):
    def payload(self, *, cpu_only, enable_gpu=False, machine_shape=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            metadata = {'id':'test/diagnostic', 'title':'Test diagnostic',
                'code_file':'test.ipynb', 'language':'python', 'kernel_type':'notebook',
                'is_private':True, 'enable_gpu':enable_gpu, 'enable_tpu':False,
                'enable_internet':False, 'machine_shape':machine_shape,
                'docker_image_pinning_type':'original', 'docker_image':'test-image'}
            (path/'kernel-metadata.json').write_text(json.dumps(metadata))
            (path/'test.ipynb').write_text(json.dumps({'cells':[]}))
            return push_payload(path, timeout_seconds=600, cpu_only=cpu_only)[0]

    def test_cpu_flag_produces_no_accelerator_request(self):
        payload = self.payload(cpu_only=True)
        self.assertFalse(payload['enableGpu'])
        self.assertFalse(payload['enableTpu'])
        self.assertIsNone(payload['machineShape'])

    def test_cpu_flag_rejects_gpu_metadata_or_shape_override(self):
        for kwargs in [{'enable_gpu':True}, {'machine_shape':'NvidiaRtxPro6000'}]:
            with self.assertRaisesRegex(ValueError, 'must not request any accelerator'):
                self.payload(cpu_only=True, **kwargs)

    def test_gpu_path_never_falls_back_to_another_accelerator(self):
        for shape in [None, 'NvidiaTeslaT4']:
            with self.assertRaisesRegex(ValueError, 'Exact validated GPU'):
                self.payload(cpu_only=False, enable_gpu=True, machine_shape=shape)


if __name__ == '__main__':
    unittest.main()
