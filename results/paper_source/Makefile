.PHONY: all paper report clean

all: paper report

paper:
	latexmk main.tex

report:
	latexmk experiment_report.tex

clean:
	latexmk -C main.tex
	latexmk -C experiment_report.tex
