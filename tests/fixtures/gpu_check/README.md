# Hardware regression captures

These fixtures replay management commands and distribution version files from
the configured backend containers. They require no GPU or vendor Python stack.
Each JSON file includes the container image, relevant installation environment,
command results, version files, expected normalized result, and text output.
JSON/YAML stdout and `--output` are checked against the same result by
`tests/test_gpu_check_live.py`.

Only commands exercised by the public check are retained. Machine identifiers,
serial numbers, UUIDs, PCI addresses, timestamps and source hashes are redacted.
An alphabetic hexadecimal bus number is preserved as a synthetic value in the
Enflame capture to cover the device-count regression.

| Vendor | Device | Count | Driver | Runtime | Runtime version |
| --- | --- | --- | --- | --- | --- |
| NVIDIA | H800 | 8 | 610.57.04 | cuda | 13.3.0 |
| Iluvatar | BI-V150 | 16 | 5.0.0 | corex | 4.5.0 |
| Hygon | BW1000 | 8 | 6.3.30-V1.4.1a | dtk | 26.04-rc4 |
| Cambricon | MLU590-M9DE | 8 | 6.2.15 | neuware | 4.7.6 |
| Moore Threads | MTT-S5000 | 8 | 5.2.0-server | musa | 5.2.0 |
| KunlunXin | P800 | 8 | 5.41.0.0 | xre | 5.37.1 |
| MetaX | C550 | 8 | 3.9.6 | maca | 3.8.1.3 |
| TsingMicro | TX8110 | 32 | 0.28791.2 | tsm | 1.4.0 |
| Enflame | ZIXIAOC200 | 8 | 1.10.6 | tops | 1.10.6 |
| Ascend | 910B4 | 8 | 26.0.rc1 | cann | 9.0.0 |
| AMD | W7900D | 8 | 6.16.13 | rocm | 7.2.3 |
| T-Head | PPU-ZW810E | 16 | 1.3.2-d7f5a2 | ppu | 2.1.0-a5f865 |

All twelve containers completed text, JSON and YAML checks successfully. Memory
usage and available indices are snapshots, not permanent hardware properties.

Ascend, AMD and T-Head were rechecked after their bastion access was restored.
CANN 9 uses the distribution manifest `share/info/runtime/version.info`; the
PPU SDK uses `release.yaml`, not the `ppu-smi` tool version. AMD's product-name
query returned a libdrm error, so its model is resolved from the exact PCI and
subsystem identity (`1002:744b`, subsystem `1002:0e0c`) verified on the W7900D
host. Unknown or mixed AMD identities remain unclassified rather than being
inferred from the backend or container name.

Sunrise hardware verification is deferred. It retains synthetic parser coverage
in `tests/test_gpu_check.py`, but has no real-output fixture from this verification.
