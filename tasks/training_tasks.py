# coding: utf-8

import unicodedata
import random
import os
import io
import shutil
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory, NamedTemporaryFile
from pathlib import Path
from more_itertools import chunked
from invoke import task
from pathvalidate import sanitize_filename
from . import utils


ROOT = Path.cwd()
TRAINEDDATA_PATH = ROOT / "traineddata"
TRAINING_FILES_PATH = ROOT / "training_files"
STAGING_AREA = ROOT / "stage"
OUTPUT_PATH = ROOT / "output"
GENERATED_FILES_PATH = STAGING_AREA / "generated_files"
LANGDATA_LSTM_PATH = TRAINEDDATA_PATH / "langdata_lstm"
PRETRAINED_MODEL_EXTRACTION_PATH = STAGING_AREA / "extracted_pretrained_model"
PROTO_MODEL_PATH = STAGING_AREA / "protomodel"
FONTS_PATH = TRAINING_FILES_PATH / "fonts"
TEXT_CORPUS_PATH = TRAINING_FILES_PATH / "text_corpus"
GENERATED_IMAGES_PLUS_TEXT_PATH = STAGING_AREA / "text_plus_images"
LSTMF_PATH = STAGING_AREA / "lstmf"
CHECKPOINT_OUTPUT_PATH = STAGING_AREA / "model_output_checkpoints"

# Misc constants
FONT_SIZES = tuple(range(10, 16))


def make_generated_path(*parts, is_dir=False, lang=""):
    newfile = GENERATED_FILES_PATH.joinpath(lang, *parts)
    if is_dir:
        newfile.mkdir(parents=True, exist_ok=True)
    else:
        newfile.parent.mkdir(parents=True, exist_ok=True)
    return newfile



def get_or_create_training_text(lang, space_separated):
    if space_separated:
        filename = make_generated_path("training.text.space.separated.txt", lang=lang)
    else:
        filename = make_generated_path("training.text.newline.separated.txt", lang=lang)
    if not filename.exists():
        text_plus_images_dir = GENERATED_IMAGES_PLUS_TEXT_PATH / lang
        txt_files = text_plus_images_dir.glob("*.txt")
        utils.combine_text_files(txt_files, output_file=filename, space_separated=space_separated)
    return filename


def generate_unicharset_from_box_files(c, output_file, box_files, box_dir, gen_func):
    """Generate unicharset from a list of box files."""
    all_boxes = [bf.name for bf in box_files]
    if sum(len(filename) for filename in all_boxes) < 8000:
        with c.cd(box_dir):
            gen_func(
                output_file,
                " ".join(all_boxes),
            )
        return
    # We need to chunk to avoid the 4k command limit on Windows
    chunks = chunked(all_boxes, n=100)
    temp_output_dir = make_generated_path("gen_unicharsets", lang=c["tesseract"]["lang"], is_dir=True)
    for (idx, chunk) in enumerate(chunks):
        temp_out_unicharset = temp_output_dir / f"{idx}.unicharset"
        with c.cd(box_dir):
            gen_func(
                temp_out_unicharset,
                " ".join(chunk),
            )
    # Now merge the generated files
    files_to_merge = " ".join(unfile.name for unfile in temp_output_dir.glob("*.unicharset"))
    with c.cd(temp_output_dir):
        c.run(
            "merge_unicharsets "
            f"{files_to_merge} "
            f"{output_file} "
        )


@task
def lang_dict(c, force=False):
    """Create a new dictionary for the language from the provided text."""
    if not c['tesseract']['new_model']:
        print("Not creating a new model. Skipping language dictionary generation...")
        return
    lang = c['tesseract']['lang']
    text_corpus = get_or_create_training_text(lang, space_separated=False)
    target_proto_dir = PROTO_MODEL_PATH / lang / lang
    target_proto_dir.mkdir(parents=True, exist_ok=True)
    c.run(
        f"create_dictdata -l {lang} "
        f"-i {text_corpus} "
        f"-d {target_proto_dir}"
    )
    print(f"Generated dictionary from  text corpus for language {lang} at {target_proto_dir}")
    # Copy specific files
    special_files_src = LANGDATA_LSTM_PATH / lang
    files_to_copy = {
        f"{lang}.config",
        "desired_characters",
        "forbidden_characters",
    }
    for filename in files_to_copy:
        src = special_files_src / filename
        if src.exists():
            shutil.copy(src, target_proto_dir)


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
            outfilename.write_text(text + "\n")
    with TemporaryDirectory() as tempdir, c.cd(output_dir), ThreadPoolExecutor(max_workers=1) as executor:
        for txt_file in output_dir.glob("*.txt"):
            for fontname in fonts_list:
                output_base = txt_file.stem.rstrip(".gt") + sanitize_filename(fontname).replace(" ", "")
                ptsize = random.choice(FONT_SIZES)
                executor.submit(
                    c.run,
                    "text2image "
                    "--xsize 2000  --ysize 320 "
                    "--min_coverage .9 --distort_image --box_padding 48 "
                    f"--margin 100 --ptsize {ptsize} "
                    f"--fontconfig_tmpdir {tempdir} "
                    f"--fonts_dir {fonts_dir} "
                    f"--text {txt_file} "
                    f'--font "{fontname}" '
                    f"--outputbase {output_base}"
                )


@task
def fontinfo(c):
    """Generate fontinfo for each font."""
    lang = c['tesseract']['lang']
    text_plus_images_dir = GENERATED_IMAGES_PLUS_TEXT_PATH / lang
    fonts_dir = FONTS_PATH / lang
    fonts_list = [
        line.strip()
        for line in (fonts_dir / "fonts.txt").read_text().split("\n")
    ]
    # Now write font properties for each font
    print("Generating font properties file for each font...")
    font_properties_dir = text_plus_images_dir / "font_properties"
    font_properties_dir.mkdir(parents=True, exist_ok=True)
    space_separated_txt = get_or_create_training_text(lang, space_separated=True)
    with c.cd(font_properties_dir), TemporaryDirectory() as tempdir, NamedTemporaryFile() as outfile:
        for fontname in fonts_list:
            print(f"Generating a list of ngrams for font: {fontname}")
            output_base = sanitize_filename(fontname.replace(" ", "_"))
            c.run(
                "text2image --render_ngrams "
                "--ptsize 32 --ligatures false"
                f"--fontconfig_tmpdir {tempdir} "
                f"--fonts_dir {fonts_dir} "
                f'--font "{fontname}" '
                f"--text {space_separated_txt} "
                f"--outputbase {output_base} "
            )
            print(f"Generating fontinfo from ngrams file for font: {fontname}")
            ngrams_file = output_base + ".box"
            c.run(
                "text2image --only_extract_font_properties "
                f"--fontconfig_tmpdir {tempdir} "
                f"--fonts_dir {fonts_dir} "
                f'--font "{fontname}" '
                f"--text {ngrams_file} "
                f"--outputbase {output_base} "
            )
    # Now copy the fontinfo files to the target folder
    fontinfo_dir = make_generated_path("fontinfo", lang=lang, is_dir=True)
    for finfo in font_properties_dir.glob("*.fontinfo"):
        shutil.copy(finfo, fontinfo_dir)


@task
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


@task(name="lstmf-list")
def write_lstmf_files_list(c):
    """Write a .txt file with all the lstmf file names."""
    lang = c['tesseract']['lang']
    lstmf_dir = LSTMF_PATH / lang
    lstmf_file_list = "\n".join(str(f.resolve()) for f in lstmf_dir.iterdir())
    print(f"Writing the list of training files to: {lstmf_dir}...")
    with open(lstmf_dir / "training.files.txt", "w", newline="\n") as file:
        file.write(lstmf_file_list)



@task
def gen_lstmf(c):
    print("Generating lstmf files from images and box files...")
    lang = c['tesseract']['lang']
    image_dir = GENERATED_IMAGES_PLUS_TEXT_PATH / lang
    tif_files = image_dir.glob("*.tif")
    lstmf_dir = LSTMF_PATH / lang
    lstmf_dir.mkdir(parents=True, exist_ok=True)
    lang_config = LANGDATA_LSTM_PATH / lang / f"{lang}.config"
    os.environ["TESSDATA_PREFIX"] = str(TRAINEDDATA_PATH)
    with c.cd(image_dir), ThreadPoolExecutor(max_workers=8) as executor:
        for tif in tif_files:
            cmd = " ".join([
                f"tesseract --psm 7 -l {lang} {tif.name}",
               f"{ tif.stem }",
               "lstm.train",
               f"{lang_config}",
            ])
            executor.submit(c.run, cmd)


@task
def mv_lstmf(c):
    """Moves lstmf files to the target folder."""
    lang = c['tesseract']['lang']
    image_dir = GENERATED_IMAGES_PLUS_TEXT_PATH / lang
    lstmf_dir = LSTMF_PATH / lang
    for lstm_file in image_dir.glob("*.lstmf"):
        shutil.move(lstm_file, lstmf_dir)


@task
def gen_unicharset(c, from_box=True):
    """Generates unicharset from box or plain text files."""
    lang = c['tesseract']['lang']
    text_plus_images_dir = GENERATED_IMAGES_PLUS_TEXT_PATH / lang
    extracted_model_directory = PRETRAINED_MODEL_EXTRACTION_PATH / lang
    proto_model_dir = PROTO_MODEL_PATH / lang
    proto_model_dir.mkdir(parents=True, exist_ok=True)
    lang_langdata_lstm_dir = LANGDATA_LSTM_PATH / lang
    final_unicharset_file = proto_model_dir / f"{lang}.unicharset"
    unicharset_norm_mode = c["tesseract"]["unicharset_norm_mode"]
    # First generate unicharset file

    def do_gen_unicharset(output_file, input_files):
        c.run(
            "unicharset_extractor "
            f"--output_unicharset { output_file } "
            f"--norm_mode {unicharset_norm_mode} "
            f"{input_files} ",
            hide=True
        )

    if from_box:
        box_files = tuple(text_plus_images_dir.glob("*.box"))
        generate_unicharset_from_box_files(
            c,
            output_file=final_unicharset_file,
            box_files=box_files,
            box_dir=text_plus_images_dir,
            gen_func=do_gen_unicharset
        )
    else:
        input_files = get_or_create_training_text(lang, space_separated=False)
        do_gen_unicharset(final_unicharset_file, input_files)
    # Merge the generated unicharset with the original unicharset file
    original_unicharset = LANGDATA_LSTM_PATH / lang / f"{lang}.unicharset"
    c.run(
        f"merge_unicharsets "
        f"{original_unicharset} "
        f"{final_unicharset_file} "
        f"{final_unicharset_file}"
    )
    lang_config_file = lang_langdata_lstm_dir / f"{ lang }.config"
    with c.cd(LANGDATA_LSTM_PATH):
        c.run(
            "set_unicharset_properties "
            f"--configfile {lang_config_file} "
            f"--U {final_unicharset_file} "
            f"--O {final_unicharset_file} "
            "--script_dir ."
        )


@task
def proto(c):
    """Generates proto model."""
    lang = c['tesseract']['lang']
    text_plus_images_dir = GENERATED_IMAGES_PLUS_TEXT_PATH / lang
    extracted_model_directory = PRETRAINED_MODEL_EXTRACTION_PATH / lang
    proto_model_dir = PROTO_MODEL_PATH / lang
    proto_model_dir.mkdir(parents=True, exist_ok=True)
    lang_langdata_lstm_dir = LANGDATA_LSTM_PATH / lang
    unicharset_file = proto_model_dir / f"{lang}.unicharset"
    is_new_model = c["tesseract"]["new_model"]
    # Finally generate the starter trainneddata file
    if is_new_model or c['tesseract']['extract_wordlist']:
        wordlist_file = proto_model_dir / lang / f"{lang}.wordlist"
    else:
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


@task
def train(c, cont=False):
    lang = c['tesseract']['lang']
    lstmf_dir = LSTMF_PATH / lang
    training_files_list = lstmf_dir / "training.files.txt"
    proto_model_dir = PROTO_MODEL_PATH / lang
    checkpoints_dir = CHECKPOINT_OUTPUT_PATH / lang
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    pretrainned_traineddata = proto_model_dir /  lang / f"{lang}.traineddata"
    old_traineddata = TRAINEDDATA_PATH / f"{lang}.traineddata"
    print(f"Training Tesseract for language: {lang}.")
    cmd = [
        "lstmtraining",
        f"--model_output {lang}",
        f"--traineddata {pretrainned_traineddata}",
        f"--train_listfile {training_files_list}",
    ]
    if cont:
        default_checkpoint_file =  checkpoints_dir / f"{lang}_checkpoint"
        cmd.extend([
            f"--continue_from {default_checkpoint_file}",
            f"--old_traineddata {old_traineddata}",
        ])
    elif not c["tesseract"]["new_model"]:
        pretrained_lstm = PRETRAINED_MODEL_EXTRACTION_PATH / lang / f"{lang}.lstm"
        cmd.extend([
            f"--continue_from {pretrained_lstm}",
            f"--old_traineddata {old_traineddata}",
        ])
    else:
        netspec_arg = '--net_spec "[1,1,0,48 Lbx256 O1c{unicharset_char_count}]"'
        num_chars_in_unichar = next((proto_model_dir / lang).glob("*.txt")).name.split("=")[1].split(".")[0]
        cmd.append(netspec_arg.format(unicharset_char_count=num_chars_in_unichar))
    with c.cd(checkpoints_dir):
        c.run(" ".join(cmd))


@task
def done(c, fast_model=False):
    lang = c['tesseract']['lang']
    checkpoint_dir = CHECKPOINT_OUTPUT_PATH / lang
    checkpoint_file =  checkpoint_dir / f"{lang}_checkpoint"
    output_model = OUTPUT_PATH / f"{lang}.traineddata"
    output_model.parent.mkdir(parents=True, exist_ok=True)
    pretrainned_traineddata = PROTO_MODEL_PATH / lang /  lang / f"{lang}.traineddata"
    cmd = [
        "lstmtraining --stop_training",
        f"--continue_from {checkpoint_file}",
        f"--traineddata {pretrainned_traineddata}",
        f"--model_output {output_model}",
    ]
    if not c["tesseract"]["new_model"]:
        old_traineddata = TRAINEDDATA_PATH / f"{lang}.traineddata"
        cmd.append(        f"--old_traineddata {old_traineddata} ",)
    if fast_model:
        cmd.append("--convert_to_int")
    c.run(" ".join(cmd))
    print(f"Done creating the new language model.\nSaved the model as: {output_model}")


@task
def evaluate(c):
    """Shows the results of lstmeval for the old and the new finetune model."""
    lang = c['tesseract']['lang']
    lstmf_dir = LSTMF_PATH / lang
    lstmf_file_list =     lstmf_dir / "training.files.txt"
    original_model = TRAINEDDATA_PATH / f"{lang}.traineddata"
    finetuned_model = OUTPUT_PATH / f"{lang}.traineddata"
    if not finetuned_model.exists():
        return print("Finetuned model was not found. Exiting...")
    cmd = " ".join([
        "lstmeval",
        f"--eval_listfile {lstmf_file_list}",
        "--model {model}",
        "--verbosity 0",
    ])
    print("Metrics for the old model")
    result = c.run(cmd.format(model=original_model))
    print("Metrics for the fine-tuned model")
    c.run(cmd.format(model=finetuned_model))


@task(name="import")
def import_files(c, from_dir, recurs=False):
    """Import box and lstmf files from a folder."""
    lang = c['tesseract']['lang']
    src_dir = Path(from_dir)
    if not src_dir.exists():
        return print(f"Folder {src_dir} not found. Exiting...")
    globber = Path.rglob if recurs else Path.glob
    task_info = [
        ("*.box", GENERATED_IMAGES_PLUS_TEXT_PATH / lang),
        ("*.gt.txt", GENERATED_IMAGES_PLUS_TEXT_PATH / lang),
        ("*.lstmf", LSTMF_PATH / lang),
    ]
    for ext, dst in task_info:
        print(f"Copying {ext} files from source ")
        dst.mkdir(parents=True, exist_ok=True)
        for file in globber(src_dir, ext):
            shutil.copy(file, dst)
    print("Done importing the data.")
