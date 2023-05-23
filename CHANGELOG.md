# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1]
Small bug fixes
- Allow URLLiberror when loading in compound classes
- Update readme with zenodo files


## [1.0.0]
### Added
- Added compound classes to sqlite file
- Compound classes are now automatically added in the library file creation pipeline. 
- Smiles are added from the specific spectrum instead of from the inchikey.

### Removed
- Compound classes cannot be added from csv file anymore. Download the newest version of the sqlite file to have compoudn classes again. 

### Changed
- Zenodo link is set to latest version instead of specific version. 

### Code structure
- Unit tests had mayor reformatting
  - Better use of global fixtures
  - Remove pickled results table files 

## [0.7.4]
- Set MS2Deepscore <2.0.0

## [0.7.3]
- Fix h5py dependency issue

## [0.7.2]
- Downloading files is more modular. 
- Loading only the models for training your own model is easier. 
- The default settings for additional metadata are changed to match mgf files from feature based molecular networking 
- Readme has been cleaned up

## [0.7.1]
- Define the newest zenodo DOI in one location. 

## [0.7.0]
- Store random forest model in Onnx format

## [0.6.6]
- Added --addional_metadat to command line usage

## [0.6.5]
- Fix dependency issue on matchmsextras

## [0.6.0]
### Added
- Make command line runnable
- Add explanation for running from command line to readme
- Check if the ionization mode of query spectra is the same as the library
- Added option to automatically split library on ionization mode.

### Changed
- When output file already exists a new file is created like output_file(1).csv
- Remove warning tensorflow
- Set tensorflow to <2.9 to prevent printing of progress bars for ms2deepscore
- Set scikit learn to version 0.24.2 to prevent risk of backwards compatibility issues.

## [0.5.7]
### Changed
- Finalize workflow for k-fold cross validation
- Add explanation for reproducing results to readme

## [0.5.6]
### Changed
- Set tensorflow version to <= 2.10.1

## [0.5.5]
### Changed
- Set tensorflow version to <= 2.4.1

## [0.5.3]
### Changed
- Set tensorflow version to <= 2.10.1

## [0.5.2]
### Changed
- Remove tensorflow warnings about feature names

## [0.5.1]
### Changed
- Remove rdkit dependency for running MS2Query 

## [0.5.0]
### Added
- Training models is now fully automatic (no need for notebooks)
- Functions for creating benchmarking results
- Functions for doing k_fold_cross_validation
- Functions for visualizing benchmarking results
### Changed
- Method for creating new library files
- Cleaning spectra functions for running are now combined with cleaning spectra functions for training

## [0.4.3]
- Do not store MS2Deepscores in results table, to prevent memory issues

## [0.4.1]

- Changed calculation of tanimoto scores, for better memory efficiency
  - Code structure changed, tanimoto scores are now calculated in create_sqlite_database, instead of library_files_creator. 

### Removed
- Option to use previously calculated tanimoto scores as input for creating the sqlite library
- 
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

[Unreleased]: https://github.com/iomega/ms2query/compare/0.7.1...HEAD
[0.6.0]: https://github.com/iomega/ms2query/compare/0.6.7...0.7.1
[0.6.0]: https://github.com/iomega/ms2query/compare/0.5.7...0.6.0
[0.5.7]: https://github.com/iomega/ms2query/compare/0.5.6...0.5.7
[0.5.6]: https://github.com/iomega/ms2query/compare/0.5.3...0.5.6
[0.5.3]: https://github.com/iomega/ms2query/compare/0.5.2...0.5.3
[0.5.2]: https://github.com/iomega/ms2query/compare/0.5.1...0.5.2
[0.5.1]: https://github.com/iomega/ms2query/compare/0.4.3...0.5.1
[0.5.0]: https://github.com/iomega/ms2query/compare/0.4.3...0.5.0
[0.4.3]: https://github.com/iomega/ms2query/compare/0.4.1...0.4.3
[0.4.1]: https://github.com/iomega/ms2query/compare/0.4.0...0.4.1
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
