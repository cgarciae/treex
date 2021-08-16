#!/bin/env bash

export FIGURE_DPI=300

streambook export example.py
jupytext --to notebook --execute example.notebook.py --output example.ipynb
rm -fr README_files
jupyter nbconvert --to markdown --output README.md example.ipynb
poetry export --without-hashes -f requirements.txt --output requirements.txt

# update readme
text=$(cat README.md)
