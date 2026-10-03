"""Model-agnostic evaluation: pick eval samples, store predictions, build comparison grids.

Any world model plugs in by writing predictions in the format of `io.write_prediction`;
model code itself lives under models/<name>/ and never depends on another model.
"""
