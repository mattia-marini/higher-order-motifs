gen-stubs:
  cd rust-core && cargo run --bin stub_gen

rust-core: gen-stubs
  uv sync --package rust-core --inexact

python-core: rust-core
  uv sync --package python-core --inexact

test-all: python-core
  uv run --package test-core all
