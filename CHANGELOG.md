# Changelog

## [3.0.0](https://github.com/corriander/vdd/compare/v2.0.2...v3.0.0) (2026-03-15)


### ⚠ BREAKING CHANGES

* add a simple cli, drop legacy python support  ([#17](https://github.com/corriander/vdd/issues/17))
* comes with dropping python 3.8-3.10 support
* **deps:** 3.11+ required

### Features

* add a simple cli ([1e41e9e](https://github.com/corriander/vdd/commit/1e41e9ee1ea07c44459e8b028f0285b4f6ace0f2))
* add a simple cli, drop legacy python support  ([#17](https://github.com/corriander/vdd/issues/17)) ([aee3e81](https://github.com/corriander/vdd/commit/aee3e81f0ee1a81c9dec1c4a413351581274f73f))


### Build System

* **deps:** drop &lt;3.11, add 3.12, 3.13; bump pandas/numpy ([1500153](https://github.com/corriander/vdd/commit/1500153db81f7b0a1c1dc751e80a6c4d87435c8d))

## [2.0.2](https://github.com/corriander/vdd/compare/v2.0.1...v2.0.2) (2026-03-15)


### Bug Fixes

* uv, update deps, fix tests ([feee20c](https://github.com/corriander/vdd/commit/feee20c75a02e17e171bf78a09aea64754a3640a))

## [2.0.1](https://github.com/corriander/vdd/compare/v2.0.0...v2.0.1) (2023-11-07)


### Bug Fixes

* remove poetry dependency ([2721f8c](https://github.com/corriander/vdd/commit/2721f8c5e34dd2ec624e0705d9e15594c8dcbb7d))

## [2.0.0](https://github.com/corriander/vdd/compare/v1.1.1...v2.0.0) (2023-11-07)


### ⚠ BREAKING CHANGES

* Supported Python range is now >=3.8,<3.12

### Bug Fixes

* pandas no longer supports out of bounds usecols ([03b258f](https://github.com/corriander/vdd/commit/03b258f0f3840f74c1ebeab1e4d4c7c5a165cce9))
* use df.loc for non-scalar assignments ([179aa8e](https://github.com/corriander/vdd/commit/179aa8e31d822250927923513abc0daf2bfc0503))


### Build System

* add support for py3.11 & pandas 2.0, drop py3.7 ([5f7e9a5](https://github.com/corriander/vdd/commit/5f7e9a51e96b9ab030fac00498ece8f156c6196b))

## [1.1.1](https://github.com/corriander/vdd/compare/v1.1.0...v1.1.1) (2023-11-07)


### Bug Fixes

* **ci:** add tag name ([481b697](https://github.com/corriander/vdd/commit/481b697196dfb789fcc37343a722a1a545be2a4c))

## [1.1.0](https://github.com/corriander/vdd/compare/1.0.0...v1.1.0) (2023-11-07)


### Features

* final version supporting 3.7 ([7d55fe4](https://github.com/corriander/vdd/commit/7d55fe4cc95710fd0f07146fb8ba6249905b117d))
