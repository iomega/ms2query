# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Changed calculation of tanimoto scores, for better memory efficiency
  - Code structure changed, tanimoto scores are now calculated in create_sqlite_database, instead of library_files_creator. 

## [0.4.0]

### Changed
- Creating your own library files for ms2query is a lot easier (see readme)
- Downloading negative mode files is added
- Downloading from zenodo is more robust

## [0.3.3]

### Changed
- Use smaller SQlite file. Tanimoto scores and peak and intensities are not stored anymore reducing the sqlite file size to 300 mb
- Generate spectrum id integers, instead of using spectrum id specified in the metadata of spectra. 

## [0.3.2]

### Added 

- Updated notebooks for performance analysis
- Solved bug in downloading library

### Changed

- Made handling different pandas versions more flexible

## [0.3.1]

### Changed

- change numbering of spectra [#135](https://github.com/iomega/ms2query/pull/135)
- updated zenodo link to new updated files [#133](https://github.com/iomega/ms2query/pull/133)

## [0.3.0]

### Added
- Switch to random forest model 
- Changed input features for random forest model [#130](https://github.com/iomega/ms2query/pull/130)
- `retention_index` and `retention_time` are cleaned by matchms in data filtering [#127](https://github.com/iomega/ms2query/pull/127)

### Changed
- switched to newer matchms (>=0.11.0) and spec2vec (>=0.6.0) versions [#127](https://github.com/iomega/ms2query/pull/127)
- Changed names for features random forest model 

### Removed
- Remove neural network model functionality
- Remove multiple chemical neighbourhood related scores

## [0.2.4]  - 2021-04-11

- Solved bug in downloading Zenodo files

## [0.2.3]  - 2021-02-11

- Added run_ms2query to make it more user friendly
- Refactored code to process spectrum per spectrum instead of list of spectra

## [0.2.2]  - 2021-09-30

- Refactored code to use results table
- Made compatible with newer pandas and gensim versions

## [0.2.1]  - 2021-06-14

### Changed

- Changed release workflow, so pip package is also updated.

## [0.2.0] - 2021-06-14

### Added

- Move library parts to Sqlite [#56](https://github.com/iomega/ms2query/pull/56)
- Define spectrum processing functions [#61](https://github.com/iomega/ms2query/pull/61)
- Extend CI workflow and add Sonarcloud [#62](https://github.com/iomega/ms2query/pull/62)
- Average inchikey score and neighbourhood score [#78](https://github.com/iomega/ms2query/pull/78) 

### Removed

- Streamlit web app (will now be future development) [#83](https://github.com/iomega/ms2query/pull/83)

### Changed

- Refactored library matching [#65](https://github.com/iomega/ms2query/pull/65)
- Split workflow into true matches and analog search [#72](https://github.com/iomega/ms2query/pull/72)
- Refactored library files creation [#74](https://github.com/iomega/ms2query/pull/74)

### Fixed


## [0.1.0] - 2021-01-01

### Added

- First ms2query prototype sketching the basic workflow and a streamlit web app.
- First test workflow and basic batches.
- Licence.

[Unreleased]: https://github.com/iomega/ms2query/compare/0.3.1...HEAD
[0.4.0]: https://github.com/iomega/ms2query/compare/0.3.3...0.4.0
[0.3.3]: https://github.com/iomega/ms2query/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/iomega/ms2query/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/iomega/ms2query/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/iomega/ms2query/compare/0.2.4...0.3.0
[0.2.4]: https://github.com/iomega/ms2query/compare/0.2.3...0.2.4
[0.2.3]: https://github.com/iomega/ms2query/compare/0.2.2...0.2.3
[0.2.2]: https://github.com/iomega/ms2query/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/iomega/ms2query/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/iomega/ms2query/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/iomega/ms2query/releases/tag/0.1.0
