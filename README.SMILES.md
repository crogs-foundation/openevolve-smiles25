# SMILES 2025 Notes

## Quick start

We recommend using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Optionally, setup pre-commit hook:

```bash
uv run pre-commit install
```

and test it

```bash
uv run pre-commit run --all-files
```

Now all python scripts can be executed as `uv run <script_name>.py`

All the necessary information can be primarly found in `README.md`

## CAD Reconstruction example

Run:

```bash
uv run openevolve-run.py ./examples/cad_holed_box/initial_program.py ./examples/cad_holed_box/evaluator.py --config ./examples/cad_holed_box/config.yaml
```

Visualize:

```bash
 uv run ./scripts/visualizer.py --path ./examples/cad_holed_box/openevolve_output/checkpoints/checkpoint_19
```

## References

- [TBD](https://tbd)
