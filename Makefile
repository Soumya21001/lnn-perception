# Makefile for the IDAI-780 Capstone Project
preprocess:
	python preprocess.py

baseline_train:
	python baseline_train.py

lnn_train:
	python lnn_train.py

evaluate:
	python evaluate.py

pipeline:
	preprocess baseline_train lnn_train evaluate
	@echo "Pipeline completed successfully!"