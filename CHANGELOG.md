# Changelog
The file contains important changes from version to version

## Unreleased

- Total rewriting of architecture;
- Add an ability to compute on GPU/CPU;


- Serialization and deserialization of neural networks to `JSON` format;
- Update error system;
- Increase speed of working;

## 0.1.3 - 16.11.2017

### Added

- Method to load data from `csv` files `neuroflow::data::DataSet`;
- Method that calculates mean values of `neuroflow::data::DataSet`;
- Method that rounds all elements of `neuroflow::data::DataSet` with given precision.

### Changed

- Documentation was rewritten;
- Improved reliability of crate;

## 0.1.2 - 13.11.2017

### Added

- Exhaustive documentation in code;

### Changed

- Functions `save` and `load` from `neuroflow::io` now returns `Result<..>`;
- Converting code to the standard.

## 0.1.1 - 12.11.2017

- First release of the crate!