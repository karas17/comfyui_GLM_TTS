import json
import os
import sys
from pathlib import Path


def load_example_texts(example_path: Path):
    data = {}
    with example_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uttid = str(obj.get("uttid"))
            syn_text = obj.get("syn_text", "")
            if uttid is not None:
                data[uttid] = syn_text
    return data


def simple_normalize(text: str) -> str:
    import re

    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。？！、,.!?；;：:”“\"'()（）\[\]【】]", "", text)
    return text


def main():
    base_dir = Path(__file__).resolve().parent
    example_path = base_dir / "examples" / "example_zh.jsonl"
    outputs_dir = base_dir / "outputs" / "pretrain_test" / "example_zh"
    out_report = outputs_dir / "asr_compare.jsonl"

    if not example_path.is_file():
        print(f"example file not found: {example_path}")
        sys.exit(1)
    if not outputs_dir.is_dir():
        print(f"outputs directory not found: {outputs_dir}")
        sys.exit(1)

    try:
        import torch
        import whisper
    except Exception as e:
        print(f"failed to import whisper or torch: {e}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = whisper.load_model("small", device=device)
    except Exception as e:
        print(f"failed to load whisper model: {e}")
        sys.exit(1)

    texts = load_example_texts(example_path)
    if not texts:
        print("no entries found in example_zh.jsonl")
        sys.exit(1)

    results = []
    for uttid, target_text in texts.items():
        wav_path = outputs_dir / f"{uttid}.wav"
        if not wav_path.is_file():
            print(f"missing audio for uttid {uttid}: {wav_path}")
            continue
        try:
            asr_result = model.transcribe(str(wav_path), language="zh")
        except Exception as e:
            print(f"whisper transcribe failed for {wav_path}: {e}")
            continue
        asr_text = asr_result.get("text", "")
        target_norm = simple_normalize(target_text)
        asr_norm = simple_normalize(asr_text)
        same = target_norm == asr_norm
        min_len = max(len(target_norm), 1)
        common = sum(1 for a, b in zip(target_norm, asr_norm) if a == b)
        char_overlap = common / min_len
        item = {
            "uttid": uttid,
            "target_text": target_text,
            "asr_text": asr_text,
            "target_norm": target_norm,
            "asr_norm": asr_norm,
            "exact_match": same,
            "char_overlap": char_overlap,
        }
        results.append(item)
        print(
            f"uttid={uttid} exact={same} overlap={char_overlap:.3f}",
            flush=True,
        )

    if not results:
        print("no ASR results produced")
        sys.exit(1)

    with out_report.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"ASR comparison written to: {out_report}")


if __name__ == "__main__":
    main()

