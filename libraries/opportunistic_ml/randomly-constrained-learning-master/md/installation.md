### Installing and building the library
#### Basic installation
- One may simply run `pip install .`.
- Optionally one may install in editable mode by specifying the `-e` flag.
This treats the repo as the installation directory and allows for local
modification of the underlying codebase.
- The script `install-dev` described below may be used for convenience.

Helper scripts:

#### Documentation
`./bin/mk-doc` builds documentation.
The generated html can then be viewed in `public/rcl.html`.
- After the module is installed, the documentation for a function can be viewed
by calling `rcl.doc.fmt_help(your_library_object_of_interest)`

#### Development installation and help
- `./bin/install-dev` checks code, `pip install`s the library in editable mode
and runs `bin/mk-doc`.
- If the optional flag `-b` or `--build` is passed then a `pylock.toml` lock
file will be generated as described below

#### Building a lockfile for the package
Running `python -m pip lock -e .` generates a new `pylock.toml` file for
the purpose of packaging the library.
