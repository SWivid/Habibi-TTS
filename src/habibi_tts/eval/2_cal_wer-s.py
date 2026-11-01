import argparse
import json
from pathlib import Path

import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

from habibi_tts.eval.utils_eval import normalize_arabic_text, word_error_rate


device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def calculate_wer(wav_dir, dialect):
    if dialect == "EGY":
        model = Wav2Vec2ForCTC.from_pretrained("IbrahimAmin/egyptian-arabic-wav2vec2-xlsr-53").to(device)
        processor = Wav2Vec2Processor.from_pretrained("IbrahimAmin/egyptian-arabic-wav2vec2-xlsr-53")
    elif dialect == "MAR":
        tokenizer = Wav2Vec2CTCTokenizer(
            "./vocab.json",
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
        processor = Wav2Vec2Processor.from_pretrained(
            "boumehdi/wav2vec2-large-xlsr-moroccan-darija", tokenizer=tokenizer
        )
        model = Wav2Vec2ForCTC.from_pretrained("boumehdi/wav2vec2-large-xlsr-moroccan-darija").to(device)
    else:
        raise ValueError(f"[Code 2_cal_wer-s.py] no available ASR model for {dialect} yet")

    benchmark = load_dataset("SWivid/Habibi", dialect, split="test")

    wer_objs = []
    for b in tqdm(benchmark):
        audio_path = b["audio"]["path"]
        gt_text = b["text"]

        try:
            waveform, sr = torchaudio.load(f"{wav_dir}/{audio_path}")
        except RuntimeError:  # hack, to eval 11labs results in mp3 format
            waveform, sr = torchaudio.load(f"{wav_dir}/{audio_path.replace('wav', 'mp3')}")
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        if dialect == "EGY":
            inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                logits = model(
                    input_values=inputs["input_values"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                ).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
        elif dialect == "MAR":
            input_values = processor(
                waveform.squeeze(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).input_values.to(device)
            logits = model(input_values).logits
            tokens = torch.argmax(logits, axis=-1)
            transcription = tokenizer.batch_decode(tokens)
        else:
            raise ValueError(f"[Code 2_cal_wer-s.py] no available ASR model for {dialect} yet")

        assert len(transcription) == 1
        pr_text = transcription[0]

        gt_text_norm = normalize_arabic_text(gt_text)
        pr_text_norm = normalize_arabic_text(pr_text)

        wer_obj = {
            "audio_path": audio_path,
            "gt_text_norm": gt_text_norm,
            "pr_text_norm": pr_text_norm,
        }
        wer = word_error_rate([pr_text_norm], [gt_text_norm])
        wer_obj["wer"] = wer
        wer_objs.append(wer_obj)

    return wer_objs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav-dir", type=str, required=True)
    parser.add_argument("-d", "--dialect", type=str, required=True, help="EGY | MAR")
    args = parser.parse_args()

    wer_objs = calculate_wer(args.wav_dir, args.dialect)

    wer_result_path = Path(args.wav_dir) / "_wer_s_results.jsonl"
    pr_text_norms = []
    gt_text_norms = []
    with open(wer_result_path, "w", encoding="utf-8") as f:
        for wer_obj in wer_objs:
            pr_text_norms.append(wer_obj["pr_text_norm"])
            gt_text_norms.append(wer_obj["gt_text_norm"])
            f.write(json.dumps(wer_obj, ensure_ascii=False) + "\n")
        f.write(f"\nGlobal WER-S: {word_error_rate(pr_text_norms, gt_text_norms)}\n")

    print(f"Global WER-S: {word_error_rate(pr_text_norms, gt_text_norms)}")
    print(f"Single WER-S results saved to {wer_result_path}")
