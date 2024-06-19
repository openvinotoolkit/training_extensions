# How to update dependencies

- Update version on `requirements.in` in the subfolder.
- Run `make <subfoler-name>`.
  - For updating dependencies of `benchmark`, run `make benchmark`.
    ```bash
    .ci/requirements$ make benchmark
    ```
  - To update all requirements.txt files
    ```bash
    .ci/requirements$ make all
    ```
