MAIN := manuscript.tex
SUPP := supplementary_material.tex
LATEXMK := latexmk
LATEXMK_FLAGS := -xelatex -synctex=1 -interaction=nonstopmode -file-line-error

.PHONY: all build clean distclean

all: build

build:
	$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN)
	$(LATEXMK) $(LATEXMK_FLAGS) $(SUPP)
	$(LATEXMK) -c $(MAIN)
	$(LATEXMK) -c $(SUPP)
	$(RM) *.synctex.gz

clean:
	$(LATEXMK) -c $(MAIN)
	$(LATEXMK) -c $(SUPP)
	$(RM) *.synctex.gz

distclean:
	$(LATEXMK) -C $(MAIN)
	$(LATEXMK) -C $(SUPP)
	$(RM) *.synctex.gz
