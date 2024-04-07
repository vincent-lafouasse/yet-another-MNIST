.PHONY: all
all:
	python3 main.py

.PHONY: format
format:
	black *.py

.PHONY: f
f: format
