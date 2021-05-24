# coding: utf-8

import shutil
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory, NamedTemporaryFile
from pathlib import Path
from invoke import task


ROOT = Path.cwd()
TRAINEDDATA_PATH = ROOT / "traineddata"
TRAINING_FILES_PATH = ROOT / "training_files"
STAGING_AREA = ROOT / "stage"
OUTPUT_PATH = ROOT / "output"
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
        extraction_directory = TRAINEDDATA_PATH / (data_file.stem)
        if extraction_directory.exists():
            shutil.rmtree(extraction_directory)
        extraction_directory.mkdir()
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
    fontslist = [l.strip() for l in (fonts_dir / "fonts.txt").read_text().split("\n")]
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
    with TemporaryDirectory() as tempdir:
        with c.cd(output_dir):
            with ThreadPoolExecutor(max_workers=1) as executor:
                for txt_file in output_dir.glob("*.txt"):
                    for idx, font in enumerate(fontslist):
                        output_base = txt_file.stem.rstrip(".gt") + f".{idx}"
                        executor.submit(
                            c.run,
                            "text2image "
                            "--xsize 1400 --ysize 256 --margin 64 "
                            f"--fontconfig_tmpdir {tempdir} "
                            f"--fonts_dir {fonts_dir} "
                            f"--text {txt_file} "
                            f'--font "{font}" '
                            f"--outputbase {output_base}"
                        )

@task(pre=(gen_images,))
def lstmf(c):
    print("Generating lstmf files from images and box files...")
    lang = c['tesseract']['lang']
    image_dir = GENERATED_IMAGES_PLUS_TEXT_PATH / lang
    tif_files = image_dir.glob("*.tif")
    lstmf_dir = LSTMF_PATH / lang
    lstmf_dir.mkdir(parents=True, exist_ok=True)
    with c.cd(lstmf_dir):
        with ThreadPoolExecutor(max_workers=8) as executor:
            for tif in tif_files:
                executor.submit(c.run, (
                    f"tesseract {tif} "
                   f"{ tif.stem } "
                   "--psm 7 lstm.train"
                ))
    # Write a list of training files to the same folder
    lstmf_file_list = "\n".join(str(f.resolve()) for f in lstmf_dir.iterdir())
    with open(lstmf_dir / "training.files.txt", "w", newline="\n") as file:
        file.write(lstmf_file_list)


@task(pre=(lstmf, extract))
def train(c):
    lang = c['tesseract']['lang']
    lstmf_dir = LSTMF_PATH / lang
    training_files_list = lstmf_dir / "training.files.txt"
    checkpoints_dir = CHECKPOINT_OUTPUT_PATH / lang
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    pretrainned_traineddata = TRAINEDDATA_PATH / f"{lang}.traineddata"
    pretrained_lstm = TRAINEDDATA_PATH / lang / f"{lang}.lstm"
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

