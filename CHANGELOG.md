# Changelog

## [3.0.3](https://github.com/corriander/vdd/compare/v3.0.2...v3.0.3) (2026-07-13)


### Bug Fixes

* **coda:** guard against requirements with no relationships ([6e321fc](https://github.com/corriander/vdd/commit/6e321fc33af57676632cb4d261b0bdf8b918dcc2))
* **coda:** make CODA.compare return a bool ([3f9e6d1](https://github.com/corriander/vdd/commit/3f9e6d16088899ea584ab32e20c5084077241a52))
* **coda:** make relationship equality type-aware ([be60fe6](https://github.com/corriander/vdd/commit/be60fe6101d2b744a7ee85eb2cb50f3139e0a075))
* **deps:** bump cryptography from 46.0.7 to 48.0.1 ([9ede717](https://github.com/corriander/vdd/commit/9ede71730fd489366a49c91a30fb3622c3cf447e))
* **deps:** bump cryptography from 46.0.7 to 48.0.1 ([#38](https://github.com/corriander/vdd/issues/38)) ([0bd491d](https://github.com/corriander/vdd/commit/0bd491d73af030a63d6c555a5b6116050a6c824b))
* merit NaN guard + hygiene batch ([#45](https://github.com/corriander/vdd/issues/45)) ([23dee42](https://github.com/corriander/vdd/commit/23dee425b71a22bf0502889d0694f0193bdf4bc2))


### Documentation

* correct google sheets API references ([92cffb5](https://github.com/corriander/vdd/commit/92cffb5eff86cdaa640e82040a3198a95044fd99))

## [3.0.2](https://github.com/corriander/vdd/compare/v3.0.1...v3.0.2) (2026-06-02)


### Bug Fixes

* **deps:** bump platformdirs from 4.9.6 to 4.10.0 ([9c69333](https://github.com/corriander/vdd/commit/9c6933301f19b119ad1e97683996bfa85050421a))
* **deps:** bump platformdirs from 4.9.6 to 4.10.0 ([#35](https://github.com/corriander/vdd/issues/35)) ([05ddcfe](https://github.com/corriander/vdd/commit/05ddcfe208f129680742388d74037d35bb5ae527))

## [3.0.1](https://github.com/corriander/vdd/compare/v3.0.0...v3.0.1) (2026-05-21)


### Bug Fixes

* **deps:** bump cryptography from 46.0.5 to 46.0.7 ([#23](https://github.com/corriander/vdd/issues/23)) ([d1af22f](https://github.com/corriander/vdd/commit/d1af22f843bda7a78a82959f2e1b7ec96a561667))
* **deps:** bump idna from 3.11 to 3.15 ([#26](https://github.com/corriander/vdd/issues/26)) ([b2bad58](https://github.com/corriander/vdd/commit/b2bad580d81da355d0e9acf2e5b43f91a6980085))
* **deps:** bump numpy from 2.4.3 to 2.4.6 ([2d8b4f4](https://github.com/corriander/vdd/commit/2d8b4f4327f54c8b2fdfe342d83d3098f42b995a))
* **deps:** bump numpy from 2.4.3 to 2.4.6 ([#30](https://github.com/corriander/vdd/issues/30)) ([5dbe3dc](https://github.com/corriander/vdd/commit/5dbe3dc874b2296f43bc3a05f825231b32abce7f))
* **deps:** bump pandas from 3.0.1 to 3.0.3 ([1a9b086](https://github.com/corriander/vdd/commit/1a9b0866f7fb31f5fd65f2ab620ec82d297e71af))
* **deps:** bump pandas from 3.0.1 to 3.0.3 ([#29](https://github.com/corriander/vdd/issues/29)) ([4311520](https://github.com/corriander/vdd/commit/4311520902ea5677a560f98af18332ddff3c9de2))
* **deps:** bump platformdirs from 4.9.4 to 4.9.6 ([988d73b](https://github.com/corriander/vdd/commit/988d73b072740ca223b6d051c501f4f69ada04cb))
* **deps:** bump platformdirs from 4.9.4 to 4.9.6 ([#31](https://github.com/corriander/vdd/issues/31)) ([912f50d](https://github.com/corriander/vdd/commit/912f50df726c041ce482f4018e2e70ace79c03a9))
* **deps:** bump pyasn1 from 0.6.2 to 0.6.3 ([#19](https://github.com/corriander/vdd/issues/19)) ([c44c902](https://github.com/corriander/vdd/commit/c44c9020812c1cfe8ebcd5c13eabd4b626971633))
* **deps:** bump pygments from 2.19.2 to 2.20.0 ([#22](https://github.com/corriander/vdd/issues/22)) ([cfa5a99](https://github.com/corriander/vdd/commit/cfa5a991f3b42d6422405b68992966b2d4129b90))
* **deps:** bump requests from 2.32.5 to 2.33.0 ([#20](https://github.com/corriander/vdd/issues/20)) ([7466b2d](https://github.com/corriander/vdd/commit/7466b2d47fda21d126cd8d7482c3a607910f2c78))
* **deps:** bump urllib3 from 2.6.3 to 2.7.0 ([#25](https://github.com/corriander/vdd/issues/25)) ([737936d](https://github.com/corriander/vdd/commit/737936df5a985147babcf0c559b20b6c0fa99270))

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
