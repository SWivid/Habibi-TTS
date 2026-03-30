# Habibi-TTS

> Official code for "Habibi: Laying the Open-Source Foundation of Unified-Dialectal Arabic Speech Synthesis"

[![arXiv](https://img.shields.io/badge/Paper-2601.13802-b31b1b.svg?logo=arXiv&style=for-the-badge)](https://arxiv.org/abs/2601.13802)
[![demo](https://img.shields.io/badge/Demo-Samples-orange.svg?logo=Github&style=for-the-badge)](https://swivid.github.io/Habibi/)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![lab](https://img.shields.io/badge/-X--LANCE-lightgrey?labelColor=grey&logo=leanpub&style=for-the-badge)](https://x-lance.sjtu.edu.cn/)
[![lab](https://img.shields.io/badge/-SII-lightgrey?labelColor=grey&logo=leanpub&style=for-the-badge)](https://www.sii.edu.cn/)    
[![hfspace](https://img.shields.io/badge/-Online%20Experience-yellow?labelColor=grey&logo=huggingface&style=for-the-badge)](https://huggingface.co/spaces/chenxie95/Habibi-TTS)
[![hfspace](https://img.shields.io/badge/-Benchmark-lightgrey?labelColor=grey&logo=huggingface&style=for-the-badge)](https://huggingface.co/datasets/SWivid/Habibi)
[![hfspace](https://img.shields.io/badge/-Model%20Suite-lightgrey?labelColor=grey&logo=huggingface&style=for-the-badge)](https://huggingface.co/SWivid/Habibi-TTS)

<div align=left>
<img src="docs/assets/main.png" width="100%">
</div>
WER-S/O: word error rates from two different ASR systems. D/S/N-MOS: dialect pronunciation accuracy, speaker similarity, and naturalness.


## Quick Start
```bash
# Install
pip install habibi-tts

# Launch the GUI TTS interface
habibi-tts_infer-gradio
```

> [!IMPORTANT]  
> Read the F5-TTS documentation for (1) [Detailed installation guidance](https://github.com/SWivid/F5-TTS?tab=readme-ov-file#installation); (2) [Best practice for inference](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/infer#inference); etc.


## CLI Usage
```bash
# Default using the Unified model (recommanded)
habibi-tts_infer-cli \
--ref_audio "assets/MSA.mp3" \
--ref_text "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم، مما سمح للجماهير بالتفاعل معه والتقاط الصور التذكارية." \
--gen_text "أهلًا، يبدو أن هناك بعض التعقيدات، لكن لا تقلق، سأرشدك بطريقة سلسة وواضحة خطوة بخطوة."

# Assign the dialect ID, rather than inferred from given reference prompt (UNK, by default)
# (best use matched dialectal content with ID: MSA, SAU, UAE, ALG, IRQ, EGY, MAR, OMN, TUN, LEV, SDN, LBY)
habibi-tts_infer-cli --dialect MSA

# Alternatively, use `.toml` file to config, see `src/habibi_tts/infer/example.toml`
habibi-tts_infer-cli -c YOUR_CUSTOM.toml

# Check more CLI features with
habibi-tts_infer-cli --help
```

> [!NOTE]  
> Some dialectal audio samples are provided under `src/habibi_tts/assets`, see the relevant [README.md](https://github.com/SWivid/Habibi-TTS/blob/main/src/habibi_tts/assets/README.md) for usage and more details.


## Training & Finetuning
See https://github.com/SWivid/Habibi-TTS/issues/2.


## Benchmarking

### 0. Benchmark setup
```bash
# Example template for benchmark use:
python src/habibi_tts/eval/0_benchmark.py -d MSA
# --dialect DIALECT (MSA | SAU | UAE | ALG | IRQ | EGY | MAR)
```

### 1. Generate benchmark samples with Habibi or 11Labs
```bash
# Zero-shot TTS performance evaluation:
accelerate launch src/habibi_tts/eval/1_infer_habibi.py -m Unified -d MAR
# --model MODEL (Unified | Specialized)
# --dialect DIALECT (MSA | SAU | UAE | ALG | IRQ | EGY | MAR)

# Use single prompt, to compare with 11Labs model:
accelerate launch src/habibi_tts/eval/1_infer_habibi.py -m Specialized -d IRQ -s
# --single (<- add this flag)

# Use single prompt, call ElevenLabs Eleven v3 (alpha) API:
pip install elevenlabs
python src/habibi_tts/eval/1_infer_11labs.py -a YOUR_API_KEY -d MSA
# --api-key API_KEY (your 11labs account API key)
# --dialect DIALECT (MSA | SAU | UAE | ALG | IRQ | EGY | MAR)
```

### 2. Transcribe samples with ASR models and calculate WER
```bash
# Evaluate WER-O with Meta Omnilingual-ASR-LLM-7B v1:
pip install omnilingual-asr
python src/habibi_tts/eval/2_cal_wer-o.py -w results/Habibi/IRQ_Specialized_single -d IRQ
# --wav-dir WAV_DIR (the folder of generated samples)
# --dialect DIALECT (MSA | SAU | UAE | ALG | IRQ | EGY | MAR)
# --batch-size BATCH_SIZE (set smaller if OOM, default 64)

# Evaluate WER-S with dialect-specific ASR models:
python src/habibi_tts/eval/2_cal_wer-s.py -w results/Habibi/MAR_Unified -d MAR
# --wav-dir WAV_DIR (the folder of generated samples)
# --dialect DIALECT (EGY | MAR)
```

### 3. Calculate speaker similarity (SIM) between generated and prompt
Download WavLM Model from [Google Drive](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view), then
```bash
python src/habibi_tts/eval/3_cal_spksim.py -w results/Habibi/MAR_Unified -d MAR -c YOUR_WAVLM_PATH
# --wav-dir WAV_DIR (the folder of generated samples)
# --dialect DIALECT (MSA | SAU | UAE | ALG | IRQ | EGY | MAR)
# --ckpt CKPT (the path of download WavLM model)

python src/habibi_tts/eval/3_cal_spksim.py -w results/Habibi/IRQ_Specialized_single -d IRQ -c YOUR_WAVLM_PATH -s
# --single (if eval single prompt or 11labs results)
```

### 4. Calculate UTMOS of generated samples
```bash
python src/habibi_tts/eval/4_cal_utmos.py -w results/11Labs_3a/MSA
# --wav-dir WAV_DIR (the folder of generated samples)
```

> [!NOTE]  
> If conflicts after omnilingual-asr installation, e.g. flash-attn, try re-install   
> `pip uninstall -y flash-attn && pip install flash-attn --no-build-isolation`


## License
All code is released under MIT License.    
The unified, SAU, and UAE models are licensed under CC-BY-NC-SA-4.0, restricted by [SADA](https://www.kaggle.com/datasets/sdaiancai/sada2022) and [Mixat](https://huggingface.co/datasets/sqrk/mixat-tri).    
The rest specialized models (ALG, EGY, IRQ, MAR, MSA) are released under Apache 2.0 license.
