"""
下記リポジトリの同名ファイルを利用させていただいております。
https://github.com/litagin02/Style-Bert-VITS2/blob/master/transcribe.py
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from torch.utils.data import Dataset
from tqdm import tqdm

import enum
from pathlib import Path

from loguru import logger

import sys
import tempfile
from typing import TextIO


class StdoutWrapper(TextIO):
    """
    `sys.stdout` wrapper for both Google Colab and local environment.
    """

    def __init__(self) -> None:
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8"
        )
        self.original_stdout = sys.stdout

    def write(self, message: str) -> int:
        result = self.temp_file.write(message)
        self.temp_file.flush()
        print(message, end="", file=self.original_stdout)
        return result

    def flush(self) -> None:
        self.temp_file.flush()

    def read(self, n: int = -1) -> str:
        self.temp_file.seek(0)
        return self.temp_file.read(n)

    def close(self) -> None:
        self.temp_file.close()

    def fileno(self) -> int:
        return self.temp_file.fileno()


try:
    import google.colab  # type: ignore

    SAFE_STDOUT = StdoutWrapper()
except ImportError:
    SAFE_STDOUT = sys.stdout


# Remove all default handlers
logger.remove()

# Add a new handler
logger.add(
    SAFE_STDOUT,
    format="<g>{time:MM-DD HH:mm:ss}</g> |<lvl>{level:^8}</lvl>| {file}:{line} | {message}",
    backtrace=True,
    diagnose=True,
)


class StrEnum(str, enum.Enum):
    """
    Enum where members are also (and must be) strings (backported from Python 3.11).
    """

    def __new__(cls, *values: str) -> "StrEnum":
        "values must already be of type `str`"
        if len(values) > 3:
            raise TypeError("too many arguments for str(): %r" % (values,))
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):  # type: ignore
                raise TypeError("%r is not a string" % (values[0],))
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):  # type: ignore
                raise TypeError("encoding must be a string, not %r" % (values[1],))
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):  # type: ignore
                raise TypeError("errors must be a string, not %r" % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()
        


# 利用可能な言語
## JP-Extra モデル利用時は JP 以外の言語の音声合成はできない
class Languages(StrEnum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"



# faster-whisperは並列処理しても速度が向上しないので、単一モデルでループ処理する
def transcribe_with_faster_whisper(
    model: "WhisperModel",
    audio_file: Path,
    initial_prompt: Optional[str] = None,
    language: str = "ja",
    num_beams: int = 1,
    no_repeat_ngram_size: int = 10,
):
    segments, _ = model.transcribe(
        str(audio_file),
        beam_size=num_beams,
        language=language,
        initial_prompt=initial_prompt,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    texts = [segment.text for segment in segments]
    return "".join(texts)


# HF pipelineで進捗表示をするために必要なDatasetクラス
class StrListDataset(Dataset[str]):
    def __init__(self, original_list: list[str]) -> None:
        self.original_list = original_list

    def __len__(self) -> int:
        return len(self.original_list)

    def __getitem__(self, i: int) -> str:
        return self.original_list[i]



# HFのWhisperはファイルリストを与えるとバッチ処理ができて速い
def transcribe_files_with_hf_whisper(
    audio_files: list[Path],
    model_id: str,
    output_file: Path,
    initial_prompt: Optional[str] = None,
    language: str = "ja",
    batch_size: int = 16,
    num_beams: int = 1,
    no_repeat_ngram_size: int = 10,
    device: str = "cuda",
    pbar: Optional[tqdm] = None,
) -> list[str]:
    import torch
    from transformers import WhisperProcessor, pipeline

    processor: WhisperProcessor = WhisperProcessor.from_pretrained(model_id)
    generate_kwargs: dict[str, Any] = {
        "language": language,
        "do_sample": False,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    logger.info(f"generate_kwargs: {generate_kwargs}")

    pipe = pipeline(
        model=model_id,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=batch_size,
        torch_dtype=torch.float16,
        device="cuda",
        trust_remote_code=True,
        # generate_kwargs=generate_kwargs,
    )
    if initial_prompt is not None:
        prompt_ids: torch.Tensor = pipe.tokenizer.get_prompt_ids(
            initial_prompt, return_tensors="pt"
        ).to(device)
        generate_kwargs["prompt_ids"] = prompt_ids

    dataset = StrListDataset([str(f) for f in audio_files])

    results: list[str] = []
    for whisper_result, file in zip(
        pipe(dataset, generate_kwargs=generate_kwargs), audio_files
    ):
        text: str = whisper_result["text"]
        # なぜかテキストの最初に" {initial_prompt}"が入るので、文字の最初からこれを削除する
        # cf. https://github.com/huggingface/transformers/issues/27594
        if text.startswith(f" {initial_prompt}"):
            text = text[len(f" {initial_prompt}") :]
        # with open(output_file, "w", encoding="utf-8") as f:
        #     for wav_file, text in zip(wav_files, results):
        #         wav_rel_path = wav_file.relative_to(input_dir)
        #         f.write(f"{wav_rel_path}|{model_name}|{language_id}|{text}\n")
        with open(output_file, "a", encoding="utf-8") as f:
            wav_rel_path = file.relative_to(input_dir)
            f.write(f"{wav_rel_path}|{model_name}|{language_id}|{text}\n")
        results.append(text)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--initial_prompt",
        type=str,
        default="こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！",
    )
    parser.add_argument(
        "--language", type=str, default="ja", choices=["ja", "en", "zh"]
    )
    parser.add_argument("--model", type=str, default="large-v3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute_type", type=str, default="bfloat16")
    parser.add_argument("--use_hf_whisper", action="store_true")
    parser.add_argument("--hf_repo_id", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=10)
    parser.add_argument("--dataset_root", type=str, default="Data")
    args = parser.parse_args()

    dataset_root = args.dataset_root

    model_name = str(args.model_name)

    dataset_root = Path(dataset_root)
    model_name = Path(model_name)

    input_dir = dataset_root / model_name / "raw"
    output_file = dataset_root / model_name / "esd.list"
    initial_prompt: str = args.initial_prompt
    initial_prompt = initial_prompt.strip('"')
    language: str = args.language
    device: str = args.device
    compute_type: str = args.compute_type
    batch_size: int = args.batch_size
    num_beams: int = args.num_beams
    no_repeat_ngram_size: int = args.no_repeat_ngram_size

    output_file.parent.mkdir(parents=True, exist_ok=True)

    wav_files = [f for f in input_dir.rglob("*.wav") if f.is_file()]
    wav_files = sorted(wav_files, key=lambda x: str(x))

    if output_file.exists():
        logger.warning(f"{output_file} exists, backing up to {output_file}.bak")
        backup_path = output_file.with_name(output_file.name + ".bak")
        if backup_path.exists():
            logger.warning(f"{output_file}.bak exists, deleting...")
            backup_path.unlink()
        output_file.rename(backup_path)

    if language == "ja":
        language_id = Languages.JP.value
    elif language == "en":
        language_id = Languages.EN.value
    elif language == "zh":
        language_id = Languages.ZH.value
    else:
        raise ValueError(f"{language} is not supported.")

    if not args.use_hf_whisper:
        from faster_whisper import WhisperModel

        logger.info(
            f"Loading faster-whisper model ({args.model}) with compute_type={compute_type}"
        )
        try:
            model = WhisperModel(args.model, device=device, compute_type=compute_type)
        except ValueError as e:
            logger.warning(f"Failed to load model, so use `auto` compute_type: {e}")
            model = WhisperModel(args.model, device=device)
        for wav_file in tqdm(wav_files, file=SAFE_STDOUT):
            text = transcribe_with_faster_whisper(
                model=model,
                audio_file=wav_file,
                initial_prompt=initial_prompt,
                language=language,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            wav_rel_path = wav_file.relative_to(input_dir)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{wav_rel_path}|{model_name}|{language_id}|{text}\n")
    else:
        if args.hf_repo_id == "":
            model_id = f"openai/whisper-{args.model}"
        else:
            model_id = args.hf_repo_id
        logger.info(f"Loading HF Whisper model ({model_id})")
        pbar = tqdm(total=len(wav_files), file=SAFE_STDOUT)
        results = transcribe_files_with_hf_whisper(
            audio_files=wav_files,
            model_id=model_id,
            initial_prompt=initial_prompt,
            language=language,
            batch_size=batch_size,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=device,
            pbar=pbar,
            output_file=output_file,
        )
        # with open(output_file, "w", encoding="utf-8") as f:
        #     for wav_file, text in zip(wav_files, results):
        #         wav_rel_path = wav_file.relative_to(input_dir)
        #         f.write(f"{wav_rel_path}|{model_name}|{language_id}|{text}\n")

    sys.exit(0)