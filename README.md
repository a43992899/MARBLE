<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="marble/utils/assets/marble-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="marble/utils/assets/marble-logo-light.svg">
    <img src="marble/utils/assets/marble-logo-light.svg" alt="Marble Logo">
  </picture>
  <br/>
  <br/>
</p>


<h3 align="center">
    <p>State-of-the-art pretrained music models for training, evaluation, inference </p>
</h3>

Marble is a modular, configuration-driven suite for training, evaluating, and performing inference on state-of-the-art pretrained music models. It leverages LightningCLI to provide easy extensibility and reproducibility.

## News and Updates
* 📌 Join Us on MIREX Discord! [<img alt="join discord" src="https://img.shields.io/discord/1379757597984296980?color=%237289da&logo=discord"/>](https://discord.gg/YxP7VkNxjk)
* **2025-06-04** Now MARBLE v2 is published on main branch! You could find the old version in `main-v1-archived` branch. 


## Key Features
1. **Modularity**: Each component—encoders, tasks, transforms, decoders—is isolated behind a common interface. You can mix and match without touching core logic.
2. **Configurability**: All experiments are driven by YAML configs. No code changes are needed to switch datasets, encoders, or training settings.
3. **Reusability**: Common routines (data loading, training loop, metrics) are implemented once in `BaseTask`, `LightningDataModule`, and shared modules.
4. **Extensibility**: Adding new encoders or tasks requires implementing a small subclass and registering it via a config.

```text
┌──────────────────┐
│ DataModule       │  yields (waveform, label, path), optional audio transforms
└─▲────────────────┘
  │
  │ waveform                     Encoded →   hidden_states[B, L, T, H]
  ▼
┌─┴───────────────┐   embedding transforms (optional)
│ Encoder         │ ────────────────────────────────────────────────────┐
└─────────────────┘                                                     │
                                                                        ▼
                                                         (LayerSelector, TimeAvgPool…)
                                                                        │
                                                                        ▼
                                      ┌─────────────────────────────────┴──┐
                                      │ Decoder(s)                         │
                                      └────────────────────────────────────┘
                                                                  │ logits
                                                                  ▼
                                                   Loss ↔ Metrics ↔ Callbacks
```


## Getting Started

1. **Install dependencies**:
    ```bash
    # 1. create a new conda env
    conda create -n marble python=3.10 -y
    conda activate marble

    # 2. install ffmpeg
    conda install -c conda-forge ffmpeg -y

    # 3. now install other dependencies
    pip install -e .

    # 4. [Optional] downgrade pip to 24.0 if you are using fairseq modules
    # pip install pip==24.0
    # pip install fairseq
    # some encoders (e.g. Xcodec) may require additional dependencies, see marble/encoders/*/requirements.txt
    ```

2. **Prepare data**: `python download.py all`.

3. **Configure**: Copy an existing YAML from `configs/` and edit paths, encoder settings, transforms, and task parameters.

4. **Run**:

   ```bash
   python cli.py fit --config configs/probe.MERT-v1-95M.GTZANGenre.yaml
   python cli.py test --config configs/probe.MERT-v1-95M.GTZANGenre.yaml
   ```

5. **Results**: Checkpoints and logs will be saved under `output/` and logged in Weights & Biases.

6. **Inference**: We provide scripts for inference on pretrained models. See the [Inference SOTA SSL MIR models](#inference-sota-ssl-mir-models) section below.


### BestRQ checkpoint probes

`scripts/run_bestrq_probe.py` evaluates frozen music-tokenizer BestRQ checkpoints on
GTZAN genre/beat, GS, EMO, or Chords1217. Every probe reduces all 24 hidden layers with
`MLPReduce(24, 1024)`; beat embeddings are interpolated to 100 Hz. Cached-front-end
checkpoints support three explicit guards:

- `--frontend-mode online`: decode waveform and compute the recipe's nnAudio 48 kHz/50 Hz Mel online;
- `--frontend-mode psnr`: compute Mel online, then apply the schema-4 -60 dB clip plus Mel PSNR40 SZ3 round-trip before the recipe's PSNR CMVN;
- `--frontend-mode sz`: never load waveform; read schema-pinned Mel `.sz` through `SpectrogramCacheDataset`.

```bash
python scripts/run_bestrq_probe.py --task GTZANGenre --lineage exp2.1 \
  --checkpoint /path/to/epoch-step=010000.ckpt \
  --recipe /path/to/mert2/recipes/bestrq/exp2.1_cached_mel16.yaml \
  --source-root /path/to/mert2 --output-root /path/to/results \
  --frontend-mode sz --spectrogram-root /path/to/marble/sz-v2 \
  --expected-contract-sha256 <sha256> \
  --spectrogram-contract-source-root /path/recorded/in/cache/contract \
  --probe-seed 1234
```

SZ mode rejects waveform transforms, validates the full immutable cache-contract SHA,
and separately validates the contract's recorded source root. The explicit
`--spectrogram-contract-source-root` permits a frozen MARBLE snapshot to resolve
the same relative dataset rows without pretending that its path is the cache's
original generation root. Use distinct output roots for frontend mode and probe
seed; an existing changed resolved config is never overwritten.


## Supported/In-coming Tasks and Encoders

<details>
<summary> 👈Click here to view the encoders and downstream tasks currently supported or under development in MARBLE.</summary>

---

### Encoders

| Name                         | Description                                                                                                  | Paper                                                         | Link                                                          |
| :--------------------------- | :----------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------ | :------------------------------------------------------------ |
| **CLaMP3**                   | Cross-Modal & Language-based Music Pretraining v3. Aligns audio, sheet music, MIDI, and multilingual text via contrastive learning. | [arXiv:2502.10362](https://arxiv.org/abs/2502.10362)          | [GitHub sanderwood/clamp3](https://github.com/sanderwood/clamp3) |
| **DaSheng**                  | Deep Audio-Signal Holistic Embeddings: masked autoencoder trained on 272 k h of diverse audio.               | [arXiv:2406.06992](https://arxiv.org/abs/2406.06992)          | [GitHub richermans/dasheng](https://github.com/richermans/dasheng) |
| **identity**                 | Pass-through encoder.                                         | —                                                             | —                                                             |
| **MERT**                     | Music understanding via large-scale self-supervised training with acoustic & musical pseudo-labels.          | [arXiv:2306.00107](https://arxiv.org/abs/2306.00107)          | [GitHub yizhilll/MERT](https://github.com/yizhilll/MERT)  
| **MuQ**                      | Self-supervised music representation with Mel Residual Vector Quantization.                                  | [arXiv:2501.01108](https://arxiv.org/abs/2501.01108)          | [GitHub Tencent-ailab/MuQ](https://github.com/tencent-ailab/MuQ) |
| **MuQMuLan**                 | Two-tower contrastive model combining MuQ audio and text for zero-shot tagging.                              | [arXiv:2501.01108](https://arxiv.org/abs/2501.01108)          | [Hugging Face OpenMuQ/MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) |
| **MusicFM**                  | Masked-token modeling in music using random projections & codebooks.                                         | [arXiv:2311.03318](https://arxiv.org/abs/2311.03318)          | [GitHub minzwon/musicfm](https://github.com/minzwon/musicfm)    |
| **Qwen2_5OmniEncoder**       | Qwen 2.5-Omni Audio Tower: a multimodal generalist model supporting text, image, audio, and video.           | [arXiv:2503.20215](https://arxiv.org/abs/2503.20215)          | [GitHub QwenLM/Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) |
| **Qwen2AudioInstructEncoder** | Instruction-tuned variant of Qwen2-Audio Encoder for interactive audio chat.                                 | [arXiv:2407.10759](https://arxiv.org/abs/2407.10759)          | [GitHub QwenLM/Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)  
| **Xcodec**                   | Improves codec semantics for audio LLMs by integrating semantic features pre-quantization.                  | [arXiv:2408.17175](https://arxiv.org/abs/2408.17175)          | [GitHub zhenye234/xcodec](https://github.com/zhenye234/xcodec)  |


---

### Tasks

[v1] tag indicates the task is implemented in MARBLE v1 and will be adapted to marble v2 soon, and [Planning] tag indicates the task is under development.

| Name                   | Description                                                                                                                                                                           | Paper                                                                                                                                                                                                                                                        | Original Link                                                                                                                                                                                             |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Chords1217**         | Large-vocabulary chord recognition on 1 217 songs drawn from Isophonics, Billboard & MARL collections.                                                                                | Mauch & Dixon (ISMIR 2019) [Large-Vocabulary Chord Recognition](https://archives.ismir.net/ismir2019/paper/000078.pdf)                                                                            | [GitHub repo](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition)                                                                                                       |
| **EMO**                | Emotion regression on EmoMusic (744 clips × 45 s; valence & arousal; R²)                                                                                                               | Bhattarai & Lee (ISMIR 2019) [Automatic Music Mood Detection Using Transfer Learning](https://pdfs.semanticscholar.org/2ec6/2d46dc43f3940b5f44deb9103c789be8b8fa.pdf)                           | [EmoMusic dataset](https://cvml.unige.ch/databases/emoMusic/)                                                                                                                                     |
| **GS**                 | Key detection on GiantSteps Key (604 EDM tracks × 2 min; 24-class major/minor; weighted accuracy ±20 cents), plus 1 077 GiantSteps-MTG-Keys for train/val.                             | Knees et al. (ISMIR 2015) [Two data sets for tempo estimation and key detection…](https://archives.ismir.net/ismir2015/paper/000246.pdf)                                                        | [GiantSteps Key dataset](https://github.com/GiantSteps/giantsteps-key-dataset)                                                                                                                   |
| **GTZANBeatTracking**  | Beat tracking on GTZAN Rhythm                                                                                                                                                         | Tzanetakis & Cook (2002) [Musical genre classification of audio signals](https://www.cs.cmu.edu/~gtzan/work/pubs/tsap02gtzan.pdf)                                                             | [Marsyas datasets](http://marsyas.info/downloads/datasets.html)                                                                                                                                   |
| **GTZANGenre**         | Genre classification on GTZAN (10 genres; 30 s clips; 930 tracks after artist-stratified “fail-filtered” split; accuracy)                                                              | Tzanetakis & Cook (2002) [Musical genre classification of audio signals](https://www.cs.cmu.edu/~gtzan/work/pubs/tsap02gtzan.pdf)                                                             | [Marsyas datasets](http://marsyas.info/downloads/datasets.html)                                                                                                                                   |
| **[Planning] HookTheoryChord**   | Chord labeling on HookTheory user-created song hooks (chord symbols).                                        | [Melody transcription via generative pre-training](https://arxiv.org/abs/2212.01884)                                                  | [GitHub repo](https://github.com/chrisdonahue/sheetsage)         |
| **HookTheoryKey**                | Key estimation on HookTheory hooks.                                                                          | [Melody transcription via generative pre-training](https://arxiv.org/abs/2212.01884)                                                  | [GitHub repo](https://github.com/chrisdonahue/sheetsage)         |
| **[Planning] HookTheoryMelody**  | Melody prediction/completion on HookTheory hooks.                                                            | [Melody transcription via generative pre-training](https://arxiv.org/abs/2212.01884)                                                  | [GitHub repo](https://github.com/chrisdonahue/sheetsage)         |
| **HookTheoryStructure**          | Structural label prediction on HookTheory hooks.                                             | [Melody transcription via generative pre-training](https://arxiv.org/abs/2212.01884)                                                  | [GitHub repo](https://github.com/chrisdonahue/sheetsage)         |
| **[Planning] HXMSA**              | Music structure analysis on the Harmonix Set (912 Western pop tracks).                                                                                    | Nieto et al. (ISMIR 2019) [The Harmonix Set: Beats, Downbeats, and Functional Segment Annotations](https://archives.ismir.net/ismir2019/paper/000068.pdf)                                      | [HarmonixSet GitHub](https://github.com/urinieto/harmonixset)                                                                                                                                    |
| **MTGGenre**           | Genre tagging on MTG-Jamendo, using split 0.                                                                                                                                   | Bogdanov et al. (ICML 2019) [The MTG-Jamendo Dataset for Automatic Music Tagging (PDF)](https://repositori.upf.edu/bitstreams/c66cf295-ad09-4576-a938-d9466a84ec48/download)                   | [MTG-Jamendo dataset](https://mtg.github.io/mtg-jamendo-dataset/)                                                                                                                                 |
| **MTGInstrument**      | Instrument tagging on MTG-Jamendo, using split 0.                                                                                                                               | Bogdanov et al. (ICML 2019) [The MTG-Jamendo Dataset for Automatic Music Tagging (PDF)](https://repositori.upf.edu/bitstreams/c66cf295-ad09-4576-a938-d9466a84ec48/download)                   | [MTG-Jamendo dataset](https://mtg.github.io/mtg-jamendo-dataset/)                                                                                                                                 |
| **MTGMood**            | Mood/theme tagging on MTG-Jamendo, using split 0.                                                                                                                                     | Bogdanov et al. (ICML 2019) [The MTG-Jamendo Dataset for Automatic Music Tagging (PDF)](https://repositori.upf.edu/bitstreams/c66cf295-ad09-4576-a938-d9466a84ec48/download)                   | [MTG-Jamendo dataset](https://mtg.github.io/mtg-jamendo-dataset/)                                                                                                                                 |
| **MTGTop50**           | Top-50 tag prediction on MTG-Jamendo, using split 0.                                                                                                                                  | Bogdanov et al. (ICML 2019) [The MTG-Jamendo Dataset for Automatic Music Tagging (PDF)](https://repositori.upf.edu/bitstreams/c66cf295-ad09-4576-a938-d9466a84ec48/download)                   | [MTG-Jamendo dataset](https://mtg.github.io/mtg-jamendo-dataset/)                                                                                                                                 |
| **MTT**                | Multi-tag auto-tagging on MagnaTagATune                                                                                                                                               | Law et al. (ISMIR 2009) [Evaluation of Algorithms Using Games: The Case of Music Tagging](https://archives.ismir.net/ismir2009/paper/000019.pdf)                                               | [MagnaTagATune dataset](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)                                                                                                              |
| **[v1] NSynth**             | Pitch-class note classification on NSynth (340 h; 4 s excerpts; 128-class; accuracy)                                                                                                  | Engel et al. (2017) [Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders](https://arxiv.org/abs/1704.01279)                                                                      | [NSynth dataset](https://magenta.withgoogle.com/nsynth)                                                                                                                                          |
| **[Planning] SHS**                | Cover-song cliques dataset: a subset of the Million Song Dataset organized into 5 854 cover groups (18 196 tracks) for cover-song identification.                                     | Bertin-Mahieux et al. (2011) [The Million Song Dataset](https://academiccommons.columbia.edu/doi/10.7916/D8377K1H/download)                                                                     | [SecondHandSongs dataset](https://millionsongdataset.com/secondhand/)                                                                                                                            |
| **[Planning] SongEval**           | Holistic song-aesthetics evaluation on 2 399 full-length songs (≈ 140 h) with 16 professional annotators across 5 aesthetic dimensions.                                                | Yao et al. (arXiv 2025) [SongEval: A Benchmark Dataset for Song Aesthetics Evaluation](https://arxiv.org/abs/2505.10793)                                                                      | [SongEval toolkit](https://github.com/ASLP-lab/SongEval)                                                                                                                                          |
| **[v1] VocalSet**           | Solo singing-voice dataset (12 vowels × 4 registers × 30 singers).                                                                                                                     | Wilkins et al. (ISMIR 2018) [VocalSet: A Singing Voice Dataset](https://interactiveaudiolab.github.io/assets/papers/wilkins_seetharaman_ismir18.pdf)                                         | [VocalSet dataset](https://zenodo.org/records/1193957)                                                                                                                                           |
| **[Planning] WildSVDD**           | Anomaly detection on WILD (Singing Voice Deepfake Detection Challenge) tracks (real vs. AI-generated).                                                                                | Zhang et al. (ICASSP 2024) [SVDD 2024: The Inaugural Singing Voice Deepfake Detection Challenge](https://arxiv.org/abs/2408.16132)                                                             | [SVDD Challenge](https://svddchallenge.org/)                                                                                                                                                     |

</details>



## Inference SOTA SSL MIR models
We are collaborating with MIREX to introduce state-of-the-art SSL-based models for Music Information Retrieval (MIR). We believe that the future of MIR lies in Self-Supervised Learning (SSL), as acquiring labeled data for MIR is costly, and fully supervised paradigms are too expensive. In contrast, the computational cost is continuously decreasing and will eventually become more affordable than manual labeling.

### Key Prediction

The `sota/predict_key.py` script performs key prediction on audio files using a pretrained model. It automatically downloads the model from Hugging Face if necessary, processes audio clips in batches, and saves the predictions (key and confidence) to a JSONL file. To run, use the following command:

```bash
python sota/predict_key.py --filelist_path <filelist> --output_path <output> --batch_size 16 --download_dir <dir>

# You may reproduce the training/testing (if you have access to corresponding data) by running 
# bash sota/reproduce_key_sota_20250618.sh
```

## Project Structure
```bash
.
├── marble/                   # Core code package
│   ├── core/                 # Base classes (BaseTask, BaseEncoder, BaseTransform)
│   ├── encoders/             # Wrapper classes for various SSL encoders
│   ├── modules/              # Shared transforms, callbacks, losses, decoders
│   ├── tasks/                # Downstream tasks (probe, few-shot, datamodules)
│   └── utils/                # IO utilities, instantiation helpers
├── cli.py                    # Entry-point for launching experiments
├── sota/                     # Scripts for state-of-the-art models and inference
├── configs/                  # Experiment configs (YAML)
├── data/                     # Datasets and metadata files
├── scripts/                  # Run scripts & utilities
├── tests/                    # Unit tests for transforms & datasets
├── pyproject.toml            # Python project metadata
└── README.md                 # This file
```

See `marble/encoders/` for available encoders. 
See `marble/tasks/` for available tasks. 




## 🚀 Adding a New Encoder

Marble supports two flexible extension modes for encoders:

### Mode 1: **Internal Extension**
1. **Implement your encoder** under `marble/encoders/`:
   ```python
   # marble/encoders/my_encoder.py
   from marble.core.base_encoder import BaseAudioEncoder

   class MyEncoder(BaseAudioEncoder):
      def __init__(self, arg1, arg2):
         super().__init__()
         # initialize your model

      def forward(self, waveforms):
         # return List[Tensor] of shape (batch, layer, seq_len, hidden_size)
         # or return a dict of representations
   ```
2. **Reference it in your YAML**:

   ```yaml
   model:
     encoder:
       class_path: marble.encoders.my_encoder.MyEncoder
       init_args:
         arg1: 123
         arg2: 456
   ```

### Mode 2: **External Extension**

1. Place `my_encoder.py` anywhere in your project (e.g. `./my_project/my_encoder.py`).
2. Use the full import path in your YAML:

   ```yaml
   model:
     encoder:
       class_path: my_project.my_encoder.MyEncoder
       init_args:
         arg1: 123
   ```

> **Optional:**
>
> * If your encoder needs embedding-level transforms, implement a `BaseEmbTransform` subclass and register under `emb_transforms`.
> * If you need custom audio preprocessing, subclass `BaseAudioTransform` and register under `audio_transforms`.

  ```yaml
  emb_transforms:
    - class_path: marble.modules.transforms.MyEmbTransform
      init_args:
        param: value

  audio_transforms:
    train:
      - class_path: marble.modules.transforms.MyAudioTransform
        init_args:
          param: value
  ```


## 🚀 Adding a New Task

Marble supports two extension modes for tasks as well:

### Mode 1: **Internal Extension**

1. **Create a new task package** under `marble/tasks/YourTask/`:

   ```
   marble/tasks/YourTask/
   ├── __init__.py
   ├── datamodule.py    # Your LightningDataModule subclass
   └── probe.py          # Your BaseTask subclass, e.g. probe, finetune, fewshot
   ```

2. **Implement your classes**:

   ```python
   # datamodule.py
   import pytorch_lightning as pl

   class YourDataModule(pl.LightningDataModule):
       def setup(self, stage=None):
           ...
       def train_dataloader(self):
           ...
       # val_dataloader, test_dataloader, etc.

   # probe.py
   from marble.core.base_task import BaseTask

   class YourTask(BaseTask):
       def __init__(self, encoder, emb_transforms, decoders, losses, metrics, sample_rate, use_ema):
           super().__init__(...)
           # custom behavior here
   ```

3. **Point your YAML** to these classes:

   ```yaml
   task:
     class_path: marble.tasks.YourTask.probe.YourTask
     init_args:
       sample_rate: 22050
       use_ema: false

   data:
     class_path: marble.tasks.YourTask.datamodule.YourDataModule
   ```

### Mode 2: **External Extension**

1. Place your task code anywhere in your project (e.g. `./my_project/probe.py`, `./my_project/datamodule.py`).
2. Reference via full import path:

   ```yaml
   model:
     class_path: my_project.probe.CustomTask

   data:
     class_path: my_project.datamodule.CustomDataModule
   ```


# Citation
```text
@article{yuan2023marble,
  title={Marble: Music audio representation benchmark for universal evaluation},
  author={Yuan, Ruibin and Ma, Yinghao and Li, Yizhi and Zhang, Ge and Chen, Xingran and Yin, Hanzhi and Liu, Yiqi and Huang, Jiawen and Tian, Zeyue and Deng, Binyue and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={39626--39647},
  year={2023}
}
```
