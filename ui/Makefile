PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

.PHONY: setup run

setup:
	cd $(PROJECT_ROOT) && uv sync

run:
	cd $(PROJECT_ROOT) && uv run streamlit run ui/streamlit_app.py
