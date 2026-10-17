"""Dataset de demos: escritura/lectura HDF5 por episodio + índice JSONL."""

from vta.data.dataset import append_index, read_episode, write_episode

__all__ = ["append_index", "read_episode", "write_episode"]
