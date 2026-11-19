## Token hashes (server `sha`, first 12 hex of the token sha256; client text sha256 in brackets)

| Prompt tokens | Kind | main drafted | main `"draft": false` | combined drafted | combined `"draft": false` | all equal |
|---|---|---|---|---|---|---|
| 52 | prose | `596a68a0fce5` [0f57311d07de] ×2 | `596a68a0fce5` [0f57311d07de] ×1 | `596a68a0fce5` [0f57311d07de] ×1 | `596a68a0fce5` [0f57311d07de] ×1 | yes |
| 68 | code | `ae1c3bc90a5f` [16d9468691fd] ×2 | `ae1c3bc90a5f` [16d9468691fd] ×1 | `ae1c3bc90a5f` [16d9468691fd] ×2 | `ae1c3bc90a5f` [16d9468691fd] ×1 | yes |
| 7,963 | prose | `50324eb0292b` [54f433d1a2ba] ×2 | `50324eb0292b` [54f433d1a2ba] ×1 | `50324eb0292b` [54f433d1a2ba] ×1 | `50324eb0292b` [54f433d1a2ba] ×1 | yes |
| 7,976 | code | `46ebf41df2f3` [bd39879d57c7] ×2 | `46ebf41df2f3` [bd39879d57c7] ×1 | `46ebf41df2f3` [bd39879d57c7] ×1 | `46ebf41df2f3` [bd39879d57c7] ×1 | yes |
| 32,540 | prose | `e5a276a9b09b` [ea226595da9e] ×2 | `e5a276a9b09b` [ea226595da9e] ×1 | `e5a276a9b09b` [ea226595da9e] ×1 | `e5a276a9b09b` [ea226595da9e] ×1 | yes |
| 32,552 | code | `348fc6e61afc` [ea1388d39cfa] ×2 | `348fc6e61afc` [ea1388d39cfa] ×1 | `348fc6e61afc` [ea1388d39cfa] ×1 | `348fc6e61afc` [ea1388d39cfa] ×1 | yes |
| 130,846 | prose | `3d2fd36f54fe` [150f345538fa] ×2 | `3d2fd36f54fe` [150f345538fa] ×1 | `3d2fd36f54fe` [150f345538fa] ×1 | `3d2fd36f54fe` [150f345538fa] ×1 | yes |
| 130,856 | code | `67b03c8a4493` [bfe9a7c59f7c] ×2 | `67b03c8a4493` [bfe9a7c59f7c] ×1 | `67b03c8a4493` [bfe9a7c59f7c] ×1 | `67b03c8a4493` [bfe9a7c59f7c] ×1 | yes |

All cells equal: True

## Decode tok/s after the first token (client, streamed), TTFT s

| Prompt tokens | Kind | Arm | main | combined | ratio | TTFT main / combined |
|---|---|---|---|---|---|---|
| 52 | prose | drafted | 103.2, 103.2 | 103.0 | 0.998 | 0.09 / 0.09 |
| 52 | prose | `"draft": false` | 32.3 | 32.3 | 1.000 | 0.09 / 0.09 |
| 68 | code | drafted | 125.3, 125.4 | 124.9, 125.2 | 0.997 | 0.13 / 0.13 |
| 68 | code | `"draft": false` | 32.3 | 32.3 | 1.001 | 0.13 / 0.12 |
| 7,963 | prose | drafted | 68.9, 69.1 | 73.2 | 1.061 | 4.63 / 4.63 |
| 7,963 | prose | `"draft": false` | 30.7 | 31.2 | 1.017 | 4.62 / 4.63 |
| 7,976 | code | drafted | 85.0, 85.0 | 90.2 | 1.060 | 4.63 / 4.63 |
| 7,976 | code | `"draft": false` | 30.7 | 31.2 | 1.017 | 4.63 / 4.62 |
| 32,540 | prose | drafted | 30.0, 30.0 | 36.6 | 1.221 | 21.55 / 21.55 |
| 32,540 | prose | `"draft": false` | 23.5 | 27.3 | 1.161 | 21.52 / 21.52 |
| 32,552 | code | drafted | 35.2, 35.2 | 42.9 | 1.220 | 21.53 / 21.55 |
| 32,552 | code | `"draft": false` | 23.5 | 27.3 | 1.161 | 21.54 / 21.53 |
| 130,846 | prose | drafted | 10.1, 10.1 | 13.2 | 1.314 | 128.78 / 128.81 |
| 130,846 | prose | `"draft": false` | 12.8 | 18.1 | 1.412 | 128.79 / 128.73 |
| 130,856 | code | drafted | 12.3, 12.3 | 16.2 | 1.314 | 128.73 / 128.77 |
| 130,856 | code | `"draft": false` | 12.8 | 18.1 | 1.412 | 128.74 / 128.74 |
