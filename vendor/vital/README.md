# vendor/vital

`init.vital` is the base preset every export is written on top of: the exporter
overwrites the parameters INSTRUMENTAL controls and leaves the rest of Vital's
structure alone.

Taken from [Syntheon](https://github.com/gudgud96/syntheon), Apache 2.0, at
`syntheon/inferencer/vital/init.vital`. It was previously referenced through a
`vendor/syntheon` gitlink that had no `.gitmodules` entry, so it could never be
fetched by a clone and every export failed with a missing file. One file
committed directly is the honest version of that dependency.
