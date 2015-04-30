
STA663_Project_Pura_John.pdf: STA663_Project_Pura_John.tex banded.png random.png
	pdflatex STA663_Project_Pura_John
	pdflatex STA663_Project_Pura_John
	pdflatex STA663_Project_Pura_John

banded.png:
	python prepare_plots.py

random.png:
	python prepare_plots.py

.PHONY: all clean allclean test

all: STA663_Project_Pura_John.pdf 

clean:
	rm -rf *csv *png *aux *log *pytxcode __pycache__ *pyc

allclean:
	make clean
	rm -f *pdf

test:
	py.test