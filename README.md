# rtod

Quantize and prune [YOLOX-Tiny](https://github.com/Megvii-BaseDetection/YOLOX), then benchmark every variant against the original on accuracy, latency and size, on CPU and GPU.

> Status: early. See [PLAN.md](PLAN.md) and [SLICES.md](SLICES.md). A Colab showcase notebook and an [indah](https://indah.abangai.dev) gallery demo are planned.

## Develop

```bash
make install && make hooks
make check
```

## Licence

Apache-2.0. YOLOX is by Megvii, also Apache-2.0; see [NOTICE](NOTICE).
