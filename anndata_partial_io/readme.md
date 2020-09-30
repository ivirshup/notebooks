These notebooks/ files contain work on reading part of an AnnData object from disk. This means:

* Reading a selection of the elements (i.e. no layers)
* Subsetting by dimension

# TODO

- [ ] Currently, dataframes are read whole and then subset
- [ ] Implement new categorical storage format
- [ ] Sparse arrays are slow to read due to problems with `h5py`'s dataset class being slow to index into. This should be solvable by using my own cache of chunks.
- [ ] Figure out api. I think `spec`s are ultimately the way to go.
- [x] Code organization. Are classes the right way to do this?
    - Probably not, since I want decoupling of methods
    - Also is a weirder way to handle extension
- [ ] Reading methods should dispatch on the container and spec
- [ ] Writing methods should dispatch on the container and object (note: this is a different container)

# Thoughts

* Potential read/ write dispatch solution

```python
# For writing
key = (type(parent), type(elem))
if key not in _REGISTRY.write:
    key = (None, type(elem))
_REGISTRY.write[key](parent, elem)

# For reading
key = (type(elem), spec)
if key not in _REGISTRY.write:
    key = (None, spec)
_REGISTRY.read[key](elem)
```

* The most important thing is that there's a stable API where all this information is passed
* How do I handle other read/ write time modifications? E.g. read dense as sparse, write sparse as dense
    * Optional/ required modifiers as frozen sets?
    * It would probably be useful to come up with other read/ write time modifications to illustrate the point.
        * "read as delayed"? Or does that warant it's own set of functions?
