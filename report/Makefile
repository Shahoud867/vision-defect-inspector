# Build the Assignment 1 report(s).
#
#   make            -> report.pdf (the 4-page submission) + supplementary.pdf
#   make report     -> report.pdf only
#   make watch      -> continuous rebuild of main.tex
#   make clean      -> remove build/ aux files
#   make distclean  -> also remove the built PDFs

LATEXMK ?= latexmk
BUILD   := build

.PHONY: all report supplementary watch clean distclean

all: report supplementary

report: main.tex $(wildcard sections/*.tex) references.bib preamble.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex
	@cp $(BUILD)/main.pdf report.pdf
	@echo "  -> report.pdf"

supplementary: supplementary.tex $(wildcard sections/*.tex) references.bib preamble.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
	@cp $(BUILD)/supplementary.pdf supplementary.pdf
	@echo "  -> supplementary.pdf"

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode main.tex

clean:
	$(LATEXMK) -C
	@rm -rf $(BUILD)

distclean: clean
	@rm -f report.pdf supplementary.pdf
