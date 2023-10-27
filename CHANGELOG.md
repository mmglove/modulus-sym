<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2023-11-xx

### Added

- Added instructions for docker build on ARM architecture.

### Changed

### Deprecated

### Removed

### Fixed

- Fixed Gradient Aggregation bug.

### Security

### Dependencies

## [1.2.0] - 2023-09-21

### Added

- Example for reservoir modeling using PINO and FNO

## [1.1.0] - 2023-08-10

### Added

- Added a CHANGELOG.md

### Removed

- Accompanying licenses (will provide in the Modulus docker image).

### Fixed

- Arch `from_config` bug for literal params.
- Fixed fused SiLU activation test.
- Update `np.bool` to `np.bool_`.
- Added a workaround fix for the CUDA graphs error in multi-node runs

### Dependencies

- Updated the base container to latest PyTorch base container which is based on torch 2.0
- Container now supports CUDA 12, Python 3.10
- Updated symengine to >=0.10.0 and vtk to >=9.2.6

## [1.0.0] - 2023-05-08

### Added

- Initial public release.