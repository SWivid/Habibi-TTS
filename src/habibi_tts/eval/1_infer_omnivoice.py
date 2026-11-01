import argparse
import multiprocessing as mp
import os
import tempfile
import random
from collections import defaultdict

import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm
from omnivoice import OmniVoice


def single_gpu_infer(gpu_id, worker_id, samples, output_dir, seed):
    torch.cuda.set_device(gpu_id)
    random.seed(seed)
    torch.manual_seed(seed)

    tts = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=f"cuda:{gpu_id}",
        dtype=torch.float16,
    )

    for ref_audio, ref_text, gen_text, gen_path in tqdm(
        samples, desc=f"GPU {gpu_id} Worker {worker_id}", position=worker_id
    ):
        file_path = os.path.join(output_dir, gen_path)

        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            continue

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".wav") as ref_file:
            ref_file.write(ref_audio)
            ref_file.flush()

            with torch.inference_mode():
                audio = tts.generate(
                    text=gen_text,
                    ref_audio=ref_file.name,
                    ref_text=ref_text,
                )
                sf.write(file_path, audio[0], 24000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dialect", required=True,
        choices=["MSA", "SAU", "UAE", "ALG", "IRQ", "EGY", "MAR"]
    )
    parser.add_argument("-g", "--gpus", type=int, nargs="+", default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=2)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    gpu_ids = args.gpus if args.gpus is not None else list(range(torch.cuda.device_count()))
    if not gpu_ids:
        raise RuntimeError("No GPUs available")
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    output_dir = f"results/OmniVoice/{args.dialect}"
    os.makedirs(output_dir, exist_ok=True)

    benchmark = load_dataset("SWivid/Habibi", args.dialect, split="test")

    spk_id_dict = defaultdict(list)
    for obj in tqdm(benchmark, desc="Preparing references"):
        ref_buf = tempfile.SpooledTemporaryFile()
        sf.write(
            ref_buf,
            obj["audio"]["array"],
            obj["audio"]["sampling_rate"],
            format="WAV",
        )
        ref_buf.seek(0)
        spk_id_dict[obj["speaker_id"]].append(
            (ref_buf.read(), obj["text"], obj["audio"]["path"])
        )
        ref_buf.close()

    # The previous entry of the same speaker is the reference; the first uses the last.
    samples = []
    for group in spk_id_dict.values():
        for i, (_, gen_text, gen_path) in enumerate(group):
            ref_audio, ref_text, _ = group[i - 1]
            samples.append((ref_audio, ref_text, gen_text, gen_path))

    samples = [
        sample for sample in samples
        if not (
            os.path.isfile(os.path.join(output_dir, sample[3]))
            and os.path.getsize(os.path.join(output_dir, sample[3])) > 0
        )
    ]

    print(f"Remaining samples: {len(samples)}")

    if not samples:
        print("All samples have been generated. Exiting.")
        return

    random.Random(args.seed).shuffle(samples)

    n_workers = len(gpu_ids) * args.batch_size
    ctx = mp.get_context("spawn")
    processes = []

    for i in range(n_workers):
        gpu_id = gpu_ids[i // args.batch_size]
        split = samples[
            i * len(samples) // n_workers:
            (i + 1) * len(samples) // n_workers
        ]

        p = ctx.Process(
            target=single_gpu_infer,
            args=(gpu_id, i, split, output_dir, args.seed + i),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError("Some GPU workers failed")


if __name__ == "__main__":
    main()
