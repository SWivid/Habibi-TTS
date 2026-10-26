import os
import sys


sys.path.append(os.getcwd())

import argparse
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from cached_path import cached_path
from datasets import load_dataset
from f5_tts.infer.utils_infer import load_model, load_vocoder
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from habibi_tts.eval.utils_eval import get_inference_prompt
from habibi_tts.infer.utils_infer import (
    cfg_strength,
    nfe_step,
    sway_sampling_coef,
    target_rms,
)


rel_path = str(files("habibi_tts").joinpath("../../"))


def single_infer(b, output_dir, ema_model, vocoder, target_sample_rate, device, seed=0):
    gen_path, ref_rms, ref_mel, ref_mel_len, total_mel_len, final_text_list = b
    ref_mel = ref_mel.to(device)
    with torch.inference_mode():
        generated, _ = ema_model.sample(
            cond=ref_mel,
            text=final_text_list,
            duration=total_mel_len,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            seed=seed,
        )
        del _
        generated = generated[:, ref_mel_len:total_mel_len, :]
        gen_mel_spec = generated.permute(0, 2, 1).to(torch.float32)
        generated_wave = vocoder.decode(gen_mel_spec).cpu()
        if ref_rms < target_rms:
            generated_wave = generated_wave * ref_rms / target_rms
        torchaudio.save(f"{output_dir}/{gen_path}", generated_wave, target_sample_rate)
    return 0


def run(accelerator):
    device = str(accelerator.device)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Unified | Specialized")
    parser.add_argument("-d", "--dialect", type=str, required=True, help="MSA | SAU | UAE | ALG | IRQ | EGY | MAR")
    parser.add_argument("-s", "--single", action="store_true", help="To compare with 11labs, single reference prompt")
    parser.add_argument("-c", "--concurrency", default=4, type=int)
    args = parser.parse_args()
    model = args.model
    dialect = args.dialect
    single = args.single

    if model == "Unified":
        ckpt_file = str(cached_path("hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors"))
        vocab_file = str(cached_path("hf://SWivid/Habibi-TTS/Unified/vocab.txt"))
        wrap_text_with_dialect_id = True
    elif model == "Specialized":
        if dialect in ["MSA", "SAU"]:
            ckpt_step = 200000
        elif dialect in ["UAE", "ALG", "IRQ", "EGY", "IRQ"]:
            ckpt_step = 100000
        else:
            raise ValueError(f"[Code 1_infer_batch.py] unexpected dialect choice: {dialect}")
        ckpt_file = str(cached_path(f"hf://SWivid/Habibi-TTS/Specialized/{dialect}/model_{ckpt_step}.safetensors"))
        vocab_file = str(cached_path(f"hf://SWivid/Habibi-TTS/Specialized/{dialect}/vocab.txt"))
        wrap_text_with_dialect_id = False
    else:
        raise ValueError(f"[Code 1_infer_batch.py] unexpected model choice: {model}")

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

    ema_model = load_model(
        model_cls, model_arc, ckpt_file, mel_spec_type=mel_spec_type, vocab_file=vocab_file, device=device
    )
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False, local_path="", device=device)

    # pull benchmark dataset
    benchmark = load_dataset("SWivid/Habibi", dialect, split="test")

    spk_id_dict = defaultdict(list)
    for obj in benchmark:
        spk_id_dict[obj["speaker_id"]].append(obj)

    # The current entry uses the previous entry with the same speaker_id as prompt
    # The last entry of a speaker_id is the prompt of the first
    benchmark = []
    for spk_id in spk_id_dict:
        for i in range(len(spk_id_dict[spk_id])):
            benchmark.append(
                [
                    spk_id_dict[spk_id][i - 1]["audio"],  # ref_audio
                    spk_id_dict[spk_id][i - 1]["text"],  # ref_text
                    spk_id_dict[spk_id][i]["text"],  # gen_text
                    spk_id_dict[spk_id][i]["audio"]["path"],  # gen_path
                    spk_id_dict[spk_id][i]["dialect"],  # gen_dialect
                ]
            )
    benchmark = get_inference_prompt(
        benchmark,
        mel_spec_kwargs=model_cfg.model.mel_spec,
        target_rms=target_rms,
        target_sample_rate=target_sample_rate,
        wrap_text_with_dialect_id=wrap_text_with_dialect_id,
        single=single,
    )
    accelerator.wait_for_everyone()

    output_dir = f"{rel_path}/results/Habibi/{dialect}_{model}{'_single' if single else ''}"
    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir)
    accelerator.wait_for_everyone()

    with accelerator.split_between_processes(benchmark) as b_split:
        executor = ProcessPoolExecutor(max_workers=args.concurrency, mp_context=mp.get_context("spawn"))
        futures = []
        for b in b_split:
            futures.append(
                executor.submit(single_infer, b, output_dir, ema_model, vocoder, target_sample_rate, device)
            )
        for futures in tqdm(
            futures, total=len(futures), disable=not accelerator.is_local_main_process, desc="Generating samples..."
        ):
            _ = futures.result()
        executor.shutdown()

    accelerator.wait_for_everyone()
    print(f"Task on {device} finished!")


def main():
    accelerator = Accelerator()

    try:
        run(accelerator)
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
