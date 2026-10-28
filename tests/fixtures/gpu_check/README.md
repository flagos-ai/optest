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

All nine containers completed text, JSON and YAML checks successfully. Memory
usage and available indices are snapshots, not permanent hardware properties.

Ascend, T-Head, AMD and Sunrise could not be reached: the bastion returned
`No found asset` for the supplied account. They have synthetic parser coverage
in `tests/test_gpu_check.py`, but no real-output fixture from this verification.
