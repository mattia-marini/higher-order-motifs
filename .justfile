gen-stubs:
  cd rust-core && cargo run --bin stub_gen

test-all:
  uv run --package test-core all

rebuild-all:
  uv sync --reinstall --package test-core

test-rust:
  cd rust-core && cargo test
