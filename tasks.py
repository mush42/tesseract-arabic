# coding: utf-8

import os
import shutil
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory, NamedTemporaryFile
from pathlib import Path
from invoke import task
from pathvalidate import sanitize_filename


ROOT = Path.cwd()
TRAINEDDATA_PATH = ROOT / "traineddata"
TRAINING_FILES_PATH = ROOT / "training_files"
STAGING_AREA = ROOT / "stage"
OUTPUT_PATH = ROOT / "output"
PRETRAINED_MODEL_EXTRACTION_PATH = STAGING_AREA / "extracted_pretrained_model"
FONTS_PATH = TRAINING_FILES_PATH / "fonts"
TEXT_CORPUS_PATH = TRAINING_FILES_PATH / "text_corpus"
GENERATED_IMAGES_PLUS_TEXT_PATH = STAGING_AREA / "text_plus_images"
LSTMF_PATH = STAGING_AREA / "lstmf"
CHECKPOINT_OUTPUT_PATH = STAGING_AREA / "model_output_checkpoints"


@task
def extract(c):
    """Extract the components of combined .traineddata language models."""
    traineddata_files = TRAINEDDATA_PATH.glob("*.traineddata")
    for data_file in traineddata_files:
        print(f"Extracting model components from {data_file}")
        lang = data_file.stem
        extraction_directory = PRETRAINED_MODEL_EXTRACTION_PATH / lang
        if extraction_directory.exists():
            shutil.rmtree(extraction_directory)
        extraction_directory.mkdir(parents=True, exist_ok=True)
        with c.cd(extraction_directory):
            c.run(
                "combine_tessdata -u "
                f"{ data_file.resolve() } "
                f"{ data_file.stem }."
            )


@task
def gen_images(c):
    lang = c['tesseract']['lang']
    output_dir = GENERATED_IMAGES_PLUS_TEXT_PATH / lang
    output_dir.mkdir(parents=True, exist_ok=True)
    fonts_dir = FONTS_PATH / lang
    fonts_list = [
        line.strip()
        for line in (fonts_dir / "fonts.txt").read_text().split("\n")
    ]
    texts_dir = TEXT_CORPUS_PATH / lang
    text_files = texts_dir.glob("*.txt")
    text_lines = {
        file.stem.strip(): {
            stripped_line
            for line in file.read_text().split("\n")
            if (stripped_line := line.strip())
        }
        for file in text_files
    }
    for (filename, lines) in text_lines.items():
        for (idx, line) in enumerate(lines):
            outfilename = output_dir / f"{filename}.{idx}.gt.txt"
            outfilename.write_text(line)
    with TemporaryDirectory() as tempdir, c.cd(output_dir), ThreadPoolExecutor(max_workers=1) as executor:
        for txt_file in output_dir.glob("*.txt"):
            for fontname in fonts_list:
                output_base = txt_file.stem.rstrip(".gt") + sanitize_filename(fontname).replace(" ", "")
                executor.submit(
                    c.run,
                    "text2image "
                    "--xsize 2400 --ysize 512 "
                    "--margin 64 --ptsize 32 "
                    f"--fontconfig_tmpdir {tempdir} "
                    f"--fonts_dir {fonts_dir} "
                    f"--text {txt_file} "
                    f'--font "{fontname}" '
                    f"--outputbase {output_base}"
                )


@task(pre=(extract, gen_images,))
def box(c):
    """Generate box files from tif images."""
    lang = c['tesseract']['lang']
    output_dir = GENERATED_IMAGES_PLUS_TEXT_PATH / lang
    tif_files = output_dir.glob("*.tif")
    with c.cd(output_dir), ThreadPoolExecutor() as executor:
        for tif in tif_files:
            box_filename = tif.with_suffix("")
            cmd = " ".join([
                f"tesseract --lang {lang}",
                str(tif),
                str(box_filename),
                "lstmbox",
            ])
            executor.submit(c.run, cmd)




@task(pre=(lstmf,))
def train(c):
    lang = c['tesseract']['lang']
    lstmf_dir = LSTMF_PATH / lang
    training_files_list = lstmf_dir / "training.files.txt"
    checkpoints_dir = CHECKPOINT_OUTPUT_PATH / lang
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    pretrainned_traineddata = TRAINEDDATA_PATH / f"{lang}.traineddata"
    pretrained_lstm = PRETRAINED_MODEL_EXTRACTION_PATH / lang / f"{lang}.lstm"
    print(f"Training Tesseract for language: {lang}.")
    with c.cd(checkpoints_dir):
        c.run(
            "lstmtraining "
            f"--model_output . "
            f"--continue_from {pretrained_lstm} "
            f"--traineddata {pretrainned_traineddata} "
            f"--train_listfile {training_files_list} "
            "--max_iterations 400"
        )


@task
def done(c):
    lang = c['tesseract']['lang']
    checkpoints_dir = CHECKPOINT_OUTPUT_PATH / lang / "._checkpoint"
    output_model = OUTPUT_PATH / f"{lang}.traineddata"
    output_model.parent.mkdir(parents=True, exist_ok=True)
    pretrainned_traineddata = TRAINEDDATA_PATH / f"{lang}.traineddata"
    c.run(
        "lstmtraining --stop_training "
        f"--continue_from {checkpoints_dir} "
        f"--traineddata {pretrainned_traineddata} "
        f"--model_output {output_model}"
    )
    print(f"Done creating the new language model.\nSaved the model as: {output_model}")

