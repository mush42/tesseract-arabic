# coding: utf-8

import unicodedata
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
LANGDATA_LSTM_PATH = TRAINEDDATA_PATH / "langdata_lstm"
PRETRAINED_MODEL_EXTRACTION_PATH = STAGING_AREA / "extracted_pretrained_model"
PROTO_MODEL_PATH = STAGING_AREA / "protomodel"
FONTS_PATH = TRAINING_FILES_PATH / "fonts"
TEXT_CORPUS_PATH = TRAINING_FILES_PATH / "text_corpus"
GENERATED_IMAGES_PLUS_TEXT_PATH = STAGING_AREA / "text_plus_images"
LSTMF_PATH = STAGING_AREA / "lstmf"
CHECKPOINT_OUTPUT_PATH = STAGING_AREA / "model_output_checkpoints"


@task
def clean(c):
    """Clean transient files."""
    print("Cleaning transient files and folders...")
    folders_to_remove = {STAGING_AREA, OUTPUT_PATH}
    for folder in {fldr for fldr in folders_to_remove if fldr.exists()}:
        print(f"Deleting folder: {folder.name}")
        shutil.rmtree(folder)


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
            outfilename = output_dir / f"{lang}.{filename}.{idx}.gt.txt"
            text = unicodedata.normalize('NFC', line)
            outfilename.write_text(text)
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
                f"tesseract --l {lang}",
                str(tif),
                str(box_filename),
                "lstmbox",
            ])
            executor.submit(c.run, cmd)


@task(pre=(extract, gen_images))
def lstmf(c):
    print("Generating lstmf files from images and box files...")
    lang = c['tesseract']['lang']
    image_dir = GENERATED_IMAGES_PLUS_TEXT_PATH / lang
    tif_files = image_dir.glob("*.tif")
    lstmf_dir = LSTMF_PATH / lang
    lstmf_dir.mkdir(parents=True, exist_ok=True)
    lang_config = PRETRAINED_MODEL_EXTRACTION_PATH / lang / f"{lang}.config"
    os.environ["TESSDATA_PREFIX"] = str(TRAINEDDATA_PATH)
    with c.cd(image_dir), ThreadPoolExecutor(max_workers=8) as executor:
        for tif in tif_files:
            cmd = " ".join([
                f"tesseract -l {lang} {tif.name}",
               f"{ tif.stem }",
               "lstm.train",
               f"{lang_config}",
            ])
            executor.submit(c.run, cmd)
    # Move .lstmf files to the lstmf_dir
    for lstm_file in image_dir.glob("*.lstmf"):
        shutil.move(lstm_file, lstmf_dir)
    # Write a list of training files to the same folder
    lstmf_file_list = "\n".join(str(f.resolve()) for f in lstmf_dir.iterdir())
    with open(lstmf_dir / "training.files.txt", "w", newline="\n") as file:
        file.write(lstmf_file_list)


@task
def proto(c):
    """Generates proto model."""
    lang = c['tesseract']['lang']
    text_plus_images_dir = GENERATED_IMAGES_PLUS_TEXT_PATH / lang
    extracted_model_directory = PRETRAINED_MODEL_EXTRACTION_PATH / lang
    proto_model_dir = PROTO_MODEL_PATH / lang
    proto_model_dir.mkdir(parents=True, exist_ok=True)
    lang_langdata_lstm_dir = LANGDATA_LSTM_PATH / lang
    # First generate unicharset file
    box_files = " ".join(str(file) for file in text_plus_images_dir.glob("*.box"))
    unicharset_file = proto_model_dir / f"{lang}.unicharset"
    c.run(
        "unicharset_extractor "
        f"--output_unicharset { unicharset_file } "
        "--norm_mode 3 "
        f"{box_files} "
    )
    # Finally generate the starter trainneddata file
    wordlist_file = lang_langdata_lstm_dir / f"{ lang }.wordlist"
    numbers_file = lang_langdata_lstm_dir  / f"{lang}.numbers"
    puncs_file = lang_langdata_lstm_dir  / f"{lang}.punc"
    cmd_components = [
        "combine_lang_model",
        f"--lang {lang}",
        f"--input_unicharset { unicharset_file }",
        "--script_dir .",
        f"--words {wordlist_file}",
        f"--numbers {numbers_file}",
        f"--puncs {puncs_file}",
        f"--output_dir {proto_model_dir}",
    ] 
    if c['tesseract']['is_rtl']:
        cmd_components.insert(1, "--lang_is_rtl")
    with c.cd(LANGDATA_LSTM_PATH):
        c.run(" ".join(cmd_components))


@task(pre=(clean, lstmf, proto))
def train(c):
    lang = c['tesseract']['lang']
    lstmf_dir = LSTMF_PATH / lang
    training_files_list = lstmf_dir / "training.files.txt"
    proto_model_dir = PROTO_MODEL_PATH / lang
    checkpoints_dir = CHECKPOINT_OUTPUT_PATH / lang
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    pretrainned_traineddata = proto_model_dir /  lang / f"{lang}.traineddata"
    pretrained_lstm = PRETRAINED_MODEL_EXTRACTION_PATH / lang / f"{lang}.lstm"
    old_traineddata = TRAINEDDATA_PATH / f"{lang}.traineddata"
    print(f"Training Tesseract for language: {lang}.")
    with c.cd(checkpoints_dir):
        c.run(
            "lstmtraining "
            f"--model_output {lang} "
            f"--continue_from {pretrained_lstm} "
            f"--traineddata {pretrainned_traineddata} "
            f"--train_listfile {training_files_list} "
            f"--old_traineddata {old_traineddata} "
        )


@task
def done(c, default_checkpoint=True, fast_model=False):
    lang = c['tesseract']['lang']
    checkpoint_dir = CHECKPOINT_OUTPUT_PATH / lang
    default_checkpoint_file =  checkpoint_dir / f"{lang}_checkpoint"
    if default_checkpoint:
        chk_files = checkpoint_dir.glob("*.checkpoint")
        keys = {}
        for ck in chk_files:
            if ck.name.count("_") == 3:
                nchars = float(ck.name.split("_")[1])
                keys[nchars] = ck
        if  keys:
            checkpoint_file = keys[min(keys)] 
        else:
            checkpoint_file = default_checkpoint_file
    else:
        checkpoint_file = default_checkpoint_file
    output_model = OUTPUT_PATH / f"{lang}.traineddata"
    output_model.parent.mkdir(parents=True, exist_ok=True)
    pretrainned_traineddata = PROTO_MODEL_PATH / lang /  lang / f"{lang}.traineddata"
    old_traineddata = TRAINEDDATA_PATH / f"{lang}.traineddata"
    cmd = [
        "lstmtraining --stop_training",
        f"--continue_from {checkpoint_file}",
        f"--traineddata {pretrainned_traineddata}",
        f"--old_traineddata {old_traineddata} ",
        f"--model_output {output_model}",
    ]
    if fast_model:
        cmd.append("--convert_to_int")
    c.run(" ".join(cmd))
    print(f"Done creating the new language model.\nSaved the model as: {output_model}")

