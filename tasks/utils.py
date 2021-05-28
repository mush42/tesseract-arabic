# coding: utf-8

from io import StringIO

def combine_text_files(txt_files, output_file, space_separated=False):
    """Join the content of given .txt files and write the result to output_file."""
    sep = " " if space_separated else "\n"
    txt_output = StringIO(newline='\n')
    for file in txt_files:
        txt_output.write(file.read_text().strip("\n").replace("\n", ""))
        txt_output.write(sep)
    with open(output_file, "w", encoding="utf-8", newline="\n") as file:
        file.write(txt_output.getvalue().strip(sep))
