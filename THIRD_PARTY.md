# Third-party code and licenses

Code written for this project is released under the MIT license in
[`LICENSE`](LICENSE). Several files are derived from other projects and keep
their original licenses, listed below. Full license texts are in
[`LICENSES/`](LICENSES).

**Non-commercial use only for NWM-derived files.** The files in the first row
are derived from [facebookresearch/nwm](https://github.com/facebookresearch/nwm)
and remain under **CC BY-NC 4.0**. Because the training, inference and
evaluation entry points depend on them, the repository as a whole can only be
used for non-commercial purposes.

| Files | Source | License | License text |
|---|---|---|---|
| `train.py`, `infer.py`, `evaluate.py`, `planning_eval.py`, `nwm/models/cdit.py`, `nwm/data/datasets.py`, `nwm/data/misc.py`, `nwm/utils/distributed.py`; `config/data_config.yaml`, `config/data_hyperparams_plan.yaml`, `config/eval_config.yaml` | [facebookresearch/nwm](https://github.com/facebookresearch/nwm), modified via [20robo/raenwm](https://github.com/20robo/raenwm) | CC BY-NC 4.0 (Meta); RAE-NWM changes MIT | [`CC-BY-NC-4.0-NWM.txt`](LICENSES/CC-BY-NC-4.0-NWM.txt), [`MIT-RAE-NWM.txt`](LICENSES/MIT-RAE-NWM.txt) |
| `data_splits/{recon,sacson,scand}/` (trajectory names only) | [facebookresearch/nwm](https://github.com/facebookresearch/nwm) and [20robo/raenwm](https://github.com/20robo/raenwm) | CC BY-NC 4.0 / MIT | as above |
| `RAE/` (stage-1 encoder/decoder, shared layers) | [bytetriper/RAE](https://github.com/bytetriper/RAE) via [20robo/raenwm](https://github.com/20robo/raenwm) | MIT | [`RAE/LICENSE`](RAE/LICENSE), [`MIT-RAE.txt`](LICENSES/MIT-RAE.txt) |
| `RAE/src/stage2/transport/` (flow-matching transport) | [willisma/SiT](https://github.com/willisma/SiT) via RAE | MIT | [`MIT-SiT.txt`](LICENSES/MIT-SiT.txt) |
| `RAE/src/stage1/decoders/decoder.py` | Hugging Face Transformers ViT-MAE, via RAE | Apache-2.0 (header in file) | header |
| `nwm/models/alignment.py` (iREPA projector and spatial normalization) | [End2End-Diffusion/iREPA](https://github.com/End2End-Diffusion/iREPA) @ `99ad4ac` | MIT | [`MIT-iREPA.txt`](LICENSES/MIT-iREPA.txt) |
| `nwm/models/gdn.py`, `nwm/models/gdn_triton.py` (adapted), `nwm/models/sana_fused_gdn.py` (unmodified copy of `diffusion/model/ops/fused_gdn.py`) | [NVlabs/Sana](https://github.com/NVlabs/Sana) @ `6298508` | Apache-2.0 | [`Apache-2.0-Sana.txt`](LICENSES/Apache-2.0-Sana.txt) |
| Multi-layer teacher for the iREPA spatial-relation objective | design follows [nanovisionx/RAEv2](https://github.com/nanovisionx/RAEv2) @ `8a0d238`; no source copied | — | — |

`data_splits/eval_indices/` (the frozen evaluation samples used in the paper)
was produced by the authors from the test splits above.

Frozen visual interfaces are loaded from their official releases and are not
redistributed here: RAE-collections (DINOv2-B/L decoders), PAE
([ZhengrongYue/PAE](https://github.com/ZhengrongYue/PAE) @ `c054d54`), Cosmos
Tokenizer CI16x16, V-JEPA 2.1
([facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) @ `204698b`)
and the S-VAE adapter from
[chandar-lab/semantic-wm](https://github.com/chandar-lab/semantic-wm) @ `2fd7618`.
The RECON, SACSoN and SCAND datasets are not redistributed. Check each
project's license before use.
