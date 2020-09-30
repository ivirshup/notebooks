These notebooks contain work on reading part of an AnnData object from disk. This means:

* Reading a selection of the elements (i.e. no layers)
* Subsetting by dimension

# TODO

* Currently, dataframes are read whole and then subset
* Sparse arrays are slow to read due to problems with `h5py`'s dataset class being slow to index into. This should be solvable by using my own cache of chunks.
* Figure out api. I think `spec`s are ultimately the way to go.
* Code organization. Are classes the right way to do this?
