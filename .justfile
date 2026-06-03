set dotenv-required := true
set dotenv-load := true 

export DATASET_DIR := absolute_path(env("DATASET_DIR"))
export PLOT_OUT_DIR := absolute_path(env("PLOT_OUT_DIR"))
export CACHE_DIR := absolute_path(env("CACHE_DIR", join(env("DATASET_DIR"), ".cache")))

export PYO3_PYTHON := absolute_path(".venv/bin/python")

gen-stubs:
  cd rust-core && cargo run --bin stub_gen

test-python:
  uv run --package test-core all

rebuild-all:
  uv sync --reinstall --package test-core

test-rust *args:
  cd rust-core/rust-core-tests && cargo run --bin {{args}}

unit-test-rust *args:
  cd rust-core/rust-core-tests && cargo test {{args}}

print-env: 
  @echo $PLOT_OUT_DIR
  @echo $DATASET_DIR
  @echo $CACHE_DIR
