
<a name="0.38.0"></a>
## [0.38.0](https://github.com/equinor/gordo-components/compare/0.38.0-rc1...0.38.0) (2019-10-29)


<a name="0.38.0-rc1"></a>
## [0.38.0-rc1](https://github.com/equinor/gordo-components/compare/0.37.0...0.38.0-rc1) (2019-10-29)

### Accept

* Accept env vars for all options

### Add

* Add required for machine-config and project-name
* Add typing_extensions to requirements
* Add catboost to requirements

### Bump

* Bump black from 19.3b0 to 19.10b0
* Bump typing-extensions from 3.7.4 to 3.7.4.1

### Compare

* Compare workflow generate stdout and file output

### Fix

* Fix black 19.3b0 to 19.10b0 format changes

### Fixed

* Fixed typo logging -> logger ([#588](https://github.com/equinor/gordo-components/issues/588))

### Move

* Move ml-server outside of influx block ([#598](https://github.com/equinor/gordo-components/issues/598))

### Remove

* Remove ruamel.yaml test dependency ([#596](https://github.com/equinor/gordo-components/issues/596))
* Remove concept of cleanup_version
* Remove Client async allow for init session ([#587](https://github.com/equinor/gordo-components/issues/587))

### Rename

* Rename kwargs to context in workflow generator CLI


<a name="0.37.0"></a>
## [0.37.0](https://github.com/equinor/gordo-components/compare/0.37.0-rc2...0.37.0) (2019-10-28)

### Add

* Add istio virtualservice to workflow ([#563](https://github.com/equinor/gordo-components/issues/563))
* Add more logging to server Add logging on the time it takes to handle X, y and the time it takes to load the model.
* Add ml-server as dependency for model-building
* Add optional base_path argument to NCS reader

### Bump

* Bump nbconvert from 5.6.0 to 5.6.1
* Bump azure-datalake-store from 0.0.47 to 0.0.48
* Bump pytest from 5.2.1 to 5.2.2
* Bump pytest-mock from 1.11.1 to 1.11.2

### Fix

* Fix grafana metrics query

### Make

* Make client have Mutex for setting endpoints ([#519](https://github.com/equinor/gordo-components/issues/519))

### Support

* Support no resampling in TimeSeriesDataset ([#539](https://github.com/equinor/gordo-components/issues/539))


<a name="0.37.0-rc2"></a>
## [0.37.0-rc2](https://github.com/equinor/gordo-components/compare/0.37.0-rc1...0.37.0-rc2) (2019-10-23)

### Add

* Add model-name to the notification-svc
* Add cli custom type for IP validation
* Add server.py func to run Gunicorn app with subprocess
* Add path for 1230-Visund
* Add dependabot badge ([#552](https://github.com/equinor/gordo-components/issues/552))
* Add and fix Client component docs ([#551](https://github.com/equinor/gordo-components/issues/551))

### Allow

* Allow parametrization of image sources in workflow ([#548](https://github.com/equinor/gordo-components/issues/548))

### Bump

* Bump numpy 1.17.2 -> 1.17.3 in pinned req
* Bump psycopg2-binary from 2.8.3 to 2.8.4

### Client

* Client CLI batch-size should match Client.__init__'s ([#545](https://github.com/equinor/gordo-components/issues/545))

### Moved

* Moved logging config to gordo_components/__init__.py

### Rename

* Rename notifier to gordo-ntf-

### Set

* Set clusterIP: None for gordo-notification-svc

### Support

* Support row filter buffering ([#542](https://github.com/equinor/gordo-components/issues/542))

### Update

* Update requirements
* Update Dockerfile entrypoint for model server
* Update cli run-server command for gunicorn
* Update DOCKER_REPO settings ([#547](https://github.com/equinor/gordo-components/issues/547))

### Url

* Url encode requests to datalake Url encode requests to allow "unsafe" chars in path names.


<a name="0.37.0-rc1"></a>
## [0.37.0-rc1](https://github.com/equinor/gordo-components/compare/0.34.0...0.37.0-rc1) (2019-10-17)

### Change

* Change live/readiness probe timeouts to server 2->5
* Change HPA target to 25%
* Change patching of time.sleep ([#543](https://github.com/equinor/gordo-components/issues/543))

### Fix

* Fix CLI connection to watchman-to-sql command ([#546](https://github.com/equinor/gordo-components/issues/546))

### Increase

* Increase testing timeout

### Move

* Move workflow generator and related parts

### Push

* Push everything but workflow-generator on CircleCI


<a name="0.34.0"></a>
## [0.34.0](https://github.com/equinor/gordo-components/compare/0.33.0...0.34.0) (2019-10-16)

### Add

* Add asset-to-path for IOC assets Path to existing IOC assets are added, including Johan Sverdrup.
* Add .readthedocs.yml ([#521](https://github.com/equinor/gordo-components/issues/521))
* Add pip requirement for dev_requirements ([#520](https://github.com/equinor/gordo-components/issues/520))

### Bump

* Bump pip-tools from 4.1.0 to 4.2.0
* Bump kubernetes from 8.0.2 to 10.0.1
* Bump aiohttp from 3.6.1 to 3.6.2

### Condense

* Condense LSTMAutoEncoder and LSTMForecast ([#510](https://github.com/equinor/gordo-components/issues/510))

### Disable

* Disable threading on server run call
* Disable CircleCI Docker layer caching

### Downgrade

* Downgrade kubernetes library to v9.0.1

### Remove

* Remove the arrow from gordo server to TimeSeriesDB in the architecture drawing
* Remove `get_model` from model.py to avoid uneccesary import `get_model` in model.py is moved to the test utils as it was not used in gordo-components.

### Support

* Support KerasRawModelRegressor ([#525](https://github.com/equinor/gordo-components/issues/525))

### Update

* Update C4 drawing with planned modules


<a name="0.33.0"></a>
## [0.33.0](https://github.com/equinor/gordo-components/compare/0.32.0-rc3...0.33.0) (2019-10-07)

### Bump

* Bump pytest from 5.2.0 to 5.2.1
* Bump pytest-mock from 1.11.0 to 1.11.1
* Bump pinned req pytz 2019.2 -> 2019.3
* Bump pinned req pyarrow 0.14.1 -> 0.15.0
* Bump pinned req jinja2 2.10.1 -> 2.10.3
* Bump pinned req asn1crypto 1.0.0 -> 1.0.1
* Bump sphinx from 1.8.5 to 2.1.2
* Bump pytest-cov from 2.7.1 to 2.8.1
* Bump recommonmark from 0.5.0 to 0.6.0
* Bump test req tornado>=4,<6 to ~6
* Bump notebook from 5.7.8 to 6.0.1 ([#489](https://github.com/equinor/gordo-components/issues/489))

### Fix

* Fix setup.py url

### Pin

* Pin keras<2.3, and update all in requirements.txt

### Remove

* Remove GET support from ML server ([#498](https://github.com/equinor/gordo-components/issues/498))
* Remove coverage from requirements.txt
* Remove unused explicit dependency protobuf
* Remove unused explicit dependency joblib

### Update

* Update dev_requirements.txt
* Update test req pytest 4.6 -> 5
* Update pip-tools and move it from req to dev_reg
* Update all in test_requirements.txt

### Upgrade

* Upgrade to tensorflow 2.0 - drop Keras ([#500](https://github.com/equinor/gordo-components/issues/500))


<a name="0.32.0-rc3"></a>
## [0.32.0-rc3](https://github.com/equinor/gordo-components/compare/0.32.0-rc2...0.32.0-rc3) (2019-10-03)

### Fix

* Fix client.io post kwarg 'files' -> 'data' ([#492](https://github.com/equinor/gordo-components/issues/492))
* Fix docstring in get_metrics_dict Input parameter ```metrics_list``` was changed to ```metrics``` in the docstring.

### Load

* Load possible functions defined in definition

### Optionally

* Optionally use parquet format in client and server ([#472](https://github.com/equinor/gordo-components/issues/472))

### Remove

* Remove docs from .dockerignore


<a name="0.32.0-rc2"></a>
## [0.32.0-rc2](https://github.com/equinor/gordo-components/compare/0.32.0-rc1...0.32.0-rc2) (2019-10-01)

### Remove

* Remove parsing of self patch-version from __init__


<a name="0.32.0-rc1"></a>
## [0.32.0-rc1](https://github.com/equinor/gordo-components/compare/0.31.0...0.32.0-rc1) (2019-09-27)

### Add

* Add new parameter evaluation_config. Remove cv_mode from the cli input and encapsulate it in the evaluation_config. The cache functionality is modified to take evaluation_config into account.

### Change

* Change retries in client to have exponential sleep time. Change from the default 5 seconds sleep to exponential sleep based on number of retries.
* Change sleep time from xor to exponentiation

### Make

* Make ML Server serve multiple models ([#460](https://github.com/equinor/gordo-components/issues/460))


<a name="0.31.0"></a>
## [0.31.0](https://github.com/equinor/gordo-components/compare/0.0.0-rc1...0.31.0) (2019-09-23)

### Add

* Add more explicit docs about resolution strings
* Add r2score, EVscore, MSE, MAE during CV ([#377](https://github.com/equinor/gordo-components/issues/377))
* Add missing space in docker_push.sh ([#454](https://github.com/equinor/gordo-components/issues/454))
* Add cv-mode to model-builder
* Add model-offset to model metadata in buidler
* Add builder._determine_offset

### Change

* Change DataProviderParam to also accept yaml from file
* Change docstring of build in cli.py to contain data-provider
* Change interactive logic. Interactive login is prioritized if set to True.

### Client

* Client adjust for any model offset ([#445](https://github.com/equinor/gordo-components/issues/445))

### Move

* Move tests to CircleCI ([#442](https://github.com/equinor/gordo-components/issues/442))

### Push

* Push coverage reports all at once

### Remove

* Remove client get/post json tests

### Upgrade

* Upgrade pip before install starts ([#455](https://github.com/equinor/gordo-components/issues/455))

### Use

* Use explicit docker registry ([#450](https://github.com/equinor/gordo-components/issues/450))


<a name="0.0.0-rc1"></a>
## [0.0.0-rc1](https://github.com/equinor/gordo-components/compare/0.30.0...0.0.0-rc1) (2019-09-06)

### Expose

* Expose `optimizer` and `optimizer_kwargs` in feedforward factory

### Make

* Make autoencoder API parameters match and add some input validation ([#431](https://github.com/equinor/gordo-components/issues/431))

### Move

* Move tests to CircleCI

### Update

* Update requirements.txt


<a name="0.30.0"></a>
## [0.30.0](https://github.com/equinor/gordo-components/compare/0.29.0...0.30.0) (2019-08-30)

### Remove

* Remove check of y being a subset of X in AnomalyView

### Support

* Support single level dataframes in serialization funcs


<a name="0.29.0"></a>
## [0.29.0](https://github.com/equinor/gordo-components/compare/0.28.0...0.29.0) (2019-08-30)

### Add

* Add **kwargs to sub_providers (iroc and ncs_reader) **kwargs passed to DataLakeProvider is also passed to sub_providers, making it possible to change default behaviour.

### Fix

* Fix overwrite of y in KerasLSTMAutoEncoder.fit ([#435](https://github.com/equinor/gordo-components/issues/435))

### Make

* Make client auto choose prediction endpoint ([#425](https://github.com/equinor/gordo-components/issues/425))

### Remove

* Remove explicit threads=None argument from DataLakeProvider __init__


<a name="0.28.0"></a>
## [0.28.0](https://github.com/equinor/gordo-components/compare/0.27.0...0.28.0) (2019-08-22)

### Add

* Add wrapper to offset adustments for scoring functions
* Add exponential retries to influx forwarder
* Add mock to the test_requirements, it was indirectly added before

### Allow

* Allow serialization of hidden GordoBase models

### Create

* Create a dev_requirements.txt ([#414](https://github.com/equinor/gordo-components/issues/414))

### Extract

* Extract model metadata from nested BaseEstimators ([#420](https://github.com/equinor/gordo-components/issues/420))

### Filter

* Filter bad data (code 0) from the datalake. Removes rows with code 0 from the data in the ncs_reader.

### Fix

* Fix logging-issue stemming from absl

### Make

* Make AutoEncoders take X and y

### NCS

* NCS reader: Default to use 1 thread

### Return

* Return None if register_dir is None in disk_registry

### Simplify

* Simplify serialization in server & client ([#410](https://github.com/equinor/gordo-components/issues/410))

### Update

* Update depencencies in test_requirements.txt
* Update dependencies in requirements.txt


<a name="0.27.0"></a>
## [0.27.0](https://github.com/equinor/gordo-components/compare/0.26.1...0.27.0) (2019-08-09)

### Bump

* Bump docker python version 3.6.8->3.6.9

### Clean

* Clean up dataset notebook example

### Use

* Use yaml.safe_load for json like parsing in CLI


<a name="0.26.1"></a>
## [0.26.1](https://github.com/equinor/gordo-components/compare/0.26.0...0.26.1) (2019-08-01)

### Fix

* Fix implicit ordering of columns in server base

### Generalize

* Generalize IROC reader (many fields) ([#375](https://github.com/equinor/gordo-components/issues/375))


<a name="0.26.0"></a>
## [0.26.0](https://github.com/equinor/gordo-components/compare/0.24.0...0.26.0) (2019-07-04)

### Add

* Add Gullfaks A as new asset ([#363](https://github.com/equinor/gordo-components/issues/363))
* Add InfImputer
* Add pytest-benchmark
* Add output activation function for feedforward NN as a parameter

### Added

* Added PERA (Peregrino) as new asset

### Allow

* Allow TimeseriesDataset to take and output target tags

### Change

* Change default keras activation function to tanh

### Document

* Document endpoints for deployed models

### Log

* Log values going into model hash key calculation
* Log expanded model configuration

### Optimize

* Optimize ML server post data processing

### Pass

* Pass kwargs to keras compiling

### Push

* Push 'latest' tag for docker images

### Support

* Support predicting n number of tags

### Upgrade

* Upgrade test requirements


<a name="0.24.0"></a>
## [0.24.0](https://github.com/equinor/gordo-components/compare/0.23.0...0.24.0) (2019-06-12)

### Add

* Add multithreaded download of tags from NCS

### Ensure

* Ensure that from and to has UTC tzone on date_range call

### Support

* Support dry-run mode that only gets some info, and returns empty frame
* Support eventually consistent endpoint metadata in Watchman


<a name="0.23.0"></a>
## [0.23.0](https://github.com/equinor/gordo-components/compare/0.25.0...0.23.0) (2019-06-07)


<a name="0.25.0"></a>
## [0.25.0](https://github.com/equinor/gordo-components/compare/0.22.0...0.25.0) (2019-06-04)

### Add

* Add output activation function for feedforward NN as a parameter
* Add multithreaded download of tags from NCS

### Added

* Added PERA (Peregrino) as new asset

### Allow

* Allow TimeseriesDataset to take and output target tags

### Autoencoders

* Autoencoders should only use X in .score()

### Change

* Change default keras activation function to tanh

### Decouple

* Decouple model input tracking and returning

### Document

* Document endpoints for deployed models

### Ensure

* Ensure that from and to has UTC tzone on date_range call

### Fix

* Fix flexible scikit-learn dependency
* Fix tests for updated scikit-learn

### Log

* Log values going into model hash key calculation
* Log expanded model configuration

### Only

* Only pass X, y to cross_val in model builder

### Refactor

* Refactor ML Server into modular model views

### Rename

* Rename auto encoders .transform() -> .predict()

### Rework

* Rework API formatting and data processing in server

### Set

* Set GordoBaseDataset.join_timeseries to use label=left

### Support

* Support predicting n number of tags
* Support dry-run mode that only gets some info, and returns empty frame
* Support eventually consistent endpoint metadata in Watchman
* Support building models without scoring/cross val
* Support parametrizing model location file

### Update

* Update README with pip install & logo placeholder

### Upgrade

* Upgrade test requirements
* Upgrade scikit-learn ~=0.21

### Use

* Use .get_params() in pipeline_into_definition


<a name="0.22.0"></a>
## [0.22.0](https://github.com/equinor/gordo-components/compare/0.21.0...0.22.0) (2019-06-04)

### Ensure

* Ensure model-config in builder is fully expanded

### Fix

* Fix setup.py license and supported python versions

### Replace

* Replace generate_window with keras functionality


<a name="0.21.0"></a>
## [0.21.0](https://github.com/equinor/gordo-components/compare/0.20.2...0.21.0) (2019-06-04)

### Add

* Add major and minor versions to model key calculation
* Add version parser to assign major, minor and patch
* Add readthedocs badge to README
* Add name as a mandatory input to model building, and insert into metadata at higher level

### Change

* Change test_requirements.in to abstract reqs

### Ensure

* Ensure model name is used when calulating cache key for model

### Rename

* Rename env variable for model name from MACHINE_NAME to MODEL_NAME

### Update

* Update .travis pypi password


<a name="0.20.2"></a>
## [0.20.2](https://github.com/equinor/gordo-components/compare/0.20.1...0.20.2) (2019-06-01)

### Add

* Add .pyup.yml config
* Add .codecov.yml config

### Client

* Client should also retry on BadRequest errors

### Integrate

* Integrate Codecov

### Pin

* Pin scikit-learn <0.21

### Push

* Push pypi packages from travis

### Upgrade

* Upgrade dependencies


<a name="0.20.1"></a>
## [0.20.1](https://github.com/equinor/gordo-components/compare/0.20.0...0.20.1) (2019-05-28)

### Change

* Change license to AGPL

### Import

* Import the TimeoutError from futures for use in Client

### Log

* Log traceback as well as exception in PredictionApiView.get_predictions

### Remove

* Remove IOC reference from architecture


<a name="0.20.0"></a>
## [0.20.0](https://github.com/equinor/gordo-components/compare/0.19.0...0.20.0) (2019-05-21)

### Add

* Add resampling_endpoint up to which data will be forward filled
* Add http status code to _handle_json IOError

### Catch

* Catch IOError and TimeoutError in POST client process

### Pass

* Pass correct start/end dates to prediction_post task

### Raise

* Raise BadRequest in _handle_json if status code matches

### Rename

* Rename Watchman's "metadata" top-level key to "endpoint-metadata"

### Support

* Support retrying in POST prediction process

### Watchman

* Watchman set endpoint healthy via metadata response


<a name="0.19.0"></a>
## [0.19.0](https://github.com/equinor/gordo-components/compare/0.18.0...0.19.0) (2019-05-14)

### Add

* Add ability to print CV scores
* Add Gina Krogh as a known NCS asset
* Add ability to have template variables in model config

### Convert

* Convert several test Keras -> PCA for speedup

### Fix

* Fix reshape args in KerasLSTMBaseEstimator::_reshape_samples

### Make

* Make KerasBaseEstimator a qualitfied ABC

### Replace

* Replace back tick with quotations
* Replace Keras model in CLI tests with PCA

### Support

* Support caching for Watchman


<a name="0.18.0"></a>
## [0.18.0](https://github.com/equinor/gordo-components/compare/0.17.0...0.18.0) (2019-05-10)

### Add

* Add support for prefix less tags

### Allow

* Allow SensorTags to be lists in JSON, and use simplejson.

### Make

* Make client optionally ignore unhealthy endpoints

### Remove

* Remove module level import of _get_dataset
* Remove globals in watchman server

### Seperate

* Seperate ambassador and current namespace in watchman


<a name="0.17.0"></a>
## [0.17.0](https://github.com/equinor/gordo-components/compare/0.16.0...0.17.0) (2019-05-03)

### Add

* Add forecast_steps to metadata in KerasLSTMForecast
* Add forecast based LSTM model + unittests
* Add pytest timeout with 100 sec timeout
* Add C4 architecture diagram
* Add reporting of 50 slowest tests

### Change

* Change running of black tests to use os.system, update black
* Change prediction log level to error

### Delete

* Delete test_model (changing directory) & fix merge conflicts

### Fix

* Fix doctests - lstm output is non deterministic

### Make

* Make KerasLSTMBaseEstimator
* Make gordo_ml_server_client fixture session scoped

### Move

* Move client utils import to avoid potential circular import

### Optimize

* Optimize use of influxdb in tests

### Refactor

* Refactor test serializers and modify generate_window
* Refactor test_azure_utils to pytest

### Remove

* Remove external httpbin.org from tests

### Test

* Test on one model kind only to speed up tests.

### Unregister

* Unregister KerasLSTMBaseEstimator & modify generate_window

### Upgrade

* Upgrade dependency pandas==0.23.4->pandas==0.24.2
* Upgrade test dependency typed-ast==1.3.4->typed-ast==1.3.5
* Upgrade test dependency ipython==7.4.0->ipython==7.5.0
* Upgrade test dependency pyrsistent==0.14.11->pyrsistent==0.15.1
* Upgrade dependency pyrsistent==0.14.11->pyrsistent==0.15.1
* Upgrade dependency sphinx-click==2.0.1->sphinx-click==2.1.0
* Upgrade dependency pip-tools==3.6.0->pip-tools==3.6.1
* Upgrade dependency grpcio==1.20.0->grpcio==1.20.1


<a name="0.16.0"></a>
## [0.16.0](https://github.com/equinor/gordo-components/compare/0.15.1...0.16.0) (2019-04-24)

### Add

* Add --forward-resampled-sensors to client predict
* Add client to Makefile
* Add Dockerfile-Client

### Convert

* Convert data_lake_provider to pytests

### Refactor

* Refactor data_provider to use influx pytest.fixture
* Refactor test_client to use influx pytest.fixture & pytests

### Specify

* Specify test directory per component in setup.cfg


<a name="0.15.1"></a>
## [0.15.1](https://github.com/equinor/gordo-components/compare/0.15.0...0.15.1) (2019-04-17)

### Make

* Make client batched POST forward correct index

### Move

* Move tests for components under gordo_components test dir

### Pytest

* Pytest fixture to check for closed event loop

### Refactor

* Refactor test_model to use pytest parametrize

### Restructure

* Restructure tests layout to match project modules

### Upgrade

* Upgrade dependencies and update urllib3 >= 1.24.2


<a name="0.15.0"></a>
## [0.15.0](https://github.com/equinor/gordo-components/compare/0.14.0...0.15.0) (2019-04-16)

### Address

* Address async Session closed on Client post/get predictions

### Convert

* Convert client forwarder logic to expect async callable

### Fix

* Fix external missing swagger

### Load

* Load model and metadata before ml server start

### Rename

* Rename DataProvider's .load_dataframes methods to .load_series

### Update

* Update all available dependencies


<a name="0.14.0"></a>
## [0.14.0](https://github.com/equinor/gordo-components/compare/0.12.1...0.14.0) (2019-03-21)

### Add

* Add default gordo client as a component
* Add pytst-asyncio async aiohttp and other recommended aio libs
* Add history to model metadata for KerasBaseEstimators
* Add history to saved models and metadata
* Add TROA to TAG_TO_PATH in ncs_reader
* Add unit tests & refactor factory functions
* Add lstm_symmetric + hourglass
* Add coverage reporting
* Add more test options for component level testing
* Add sphinx documentation
* Add score for LSTM autoencoder
* Add RandomDataProvider
* Add dataset/pandas filter_rows

### Address

* Address new mypy 0.7 errors

### Allow

* Allow InfluxDataProvider to take a client directly

### Bump

* Bump notebook test dependency to 5.7.8

### Enable

* Enable travis parallel builds
* Enable pip caching on travis

### Filter

* Filter deprecation warnings in pytest.ini

### Find

* Find GordoBase and fetch metadata, update tests with more model configs

### Generalize

* Generalize kwargs in models

### IROC

* IROC reader: ignore empty files  ([#171](https://github.com/equinor/gordo-components/issues/171))

### Integrate

* Integrate row_filter into TimeSeriesDataset

### LSTM

* LSTM Autoencoder

### Make

* Make GordoServerBaseTestCase for use in other tests
* Make Watchman report status & logs of model builders / servers
* Make requirements.in files for test and install

### Optionally

* Optionally register and re-use old models

### Parameterize

* Parameterize influxdatabase context

### Pass

* Pass namespace parameter to watchman server

### Refactor

* Refactor server checking of start & end date params

### Remove

* Remove outdated module level READMEs

### Set

* Set default azure log level to INFO

### Support

* Support .to_dict() for all DataProviders
* Support InfluxDataProvider client creation from uri

### Update

* Update .dockerignore
* Update pytest config with pytest.ini for defaults


<a name="0.12.1"></a>
## [0.12.1](https://github.com/equinor/gordo-components/compare/0.12.0...0.12.1) (2019-03-08)

### Fast

* Fast quit IROC reader on empty tag-list


<a name="0.12.0"></a>
## [0.12.0](https://github.com/equinor/gordo-components/compare/0.13.0...0.12.0) (2019-03-07)


<a name="0.13.0"></a>
## [0.13.0](https://github.com/equinor/gordo-components/compare/0.10.0...0.13.0) (2019-03-04)

### Add

* Add score for LSTM autoencoder
* Add RandomDataProvider
* Add dataset/pandas filter_rows
* Add IROC reader to DataLakeBackedDataset
* Add /prediction GET route to gordo ml server
* Add pytest-xdist for parallel testing
* Add cross validation to build_model process w/ metadata
* Add .score(...) abstract method to GordoBase

### Enable

* Enable travis parallel builds
* Enable pip caching on travis

### Fast

* Fast quit IROC reader on empty tag-list

### Fix

* Fix missing abc.ABC parent in GordoBase

### Implement

* Implement .score(...) method for KerasAutoEncoder
* Implement DataProvider and Dataset separation

### Integrate

* Integrate row_filter into TimeSeriesDataset

### LSTM

* LSTM Autoencoder

### Make

* Make pytest use all cores, not always 4

### Move

* Move pytest-xdist to test_requirements

### Support

* Support ability to download model from server

### Update

* Update most dependencies with pip-compile


<a name="0.10.0"></a>
## [0.10.0](https://github.com/equinor/gordo-components/compare/0.11.0...0.10.0) (2019-02-22)


<a name="0.11.0"></a>
## [0.11.0](https://github.com/equinor/gordo-components/compare/0.9.0...0.11.0) (2019-02-22)

### Add

* Add model constructor feedforward_hourglass
* Add model constructor feedforward_symmetric
* Add requirement numexpr as recommended by pandas

### Change

* Change default KerasAutoEncoder kind to hourglass

### Fix

* Fix types in register_model_builder

### Implement

* Implement DataProvider and Dataset separation

### Rename

* Rename feedforward_symetric to feedforward_model


<a name="0.9.0"></a>
## [0.9.0](https://github.com/equinor/gordo-components/compare/0.8.0...0.9.0) (2019-02-21)

### Update

* Update cli.build to reference data config keys and not env vars


<a name="0.8.0"></a>
## [0.8.0](https://github.com/equinor/gordo-components/compare/0.7.0...0.8.0) (2019-02-18)

### Add

* Add example notebooks for Gordo use locally
* Add Travis build badge to README

### Change

* Change version in metadata to gordo-server-version

### Make

* Make .builder.build_model always return model & metadata
* Make gordo_components.dataset.datasets public
* Make gordo_components.dataset.dataset.get_dataset private

### Remove

* Remove TestFail notebook and associated test

### Rename

* Rename /predictions to /prediction for gordo server

### Update

* Update example notebook to import from public .datasets
* Update README with link to example notebooks


<a name="0.7.0"></a>
## [0.7.0](https://github.com/equinor/gordo-components/compare/0.6.1...0.7.0) (2019-02-10)

### Add

* Add DataLakeBackedDataset as a possible data getter
* Add testnodocker to run tests without requiring docker on machine
* Add pytest-flakes to unittests

### Change

* Change default model builder dataset type to DataLakeBackedDataset

### Extract

* Extract the preprocessing and joining to a non-datalake aware function

### Make

* Make Watchman return metadata from servers & report naked endpoints

### Pickup

* Pickup METADATA passed in from infrastructure

### Remove

* Remove unused imports

### Support

* Support dumps and loads for gordo serializer

### Update

* Update ambassador host to match namespace it's in
* Update returned metadata from Gordo Server
* Update watchman to just use target names
* Update serializer.loads/dumps to use buffer for tarfile


<a name="0.6.1"></a>
## [0.6.1](https://github.com/equinor/gordo-components/compare/0.6.0...0.6.1) (2019-01-22)

### Fix

* Fix static assets for swagger & improve swagger UI


<a name="0.6.0"></a>
## [0.6.0](https://github.com/equinor/gordo-components/compare/0.5.0...0.6.0) (2019-01-21)

### Add

* Add Swagger API docs
* Add black formatter to project

### Black

* Black formatting of setup.py

### Make

* Make formatting test case running path insensitive
* Make cli.py a runnable file

### More

* More dynamic instantiation of dataset objects

### Move

* Move black formatting into unittests

### Rename

* Rename test config file to more generic name


<a name="0.5.0"></a>
## [0.5.0](https://github.com/equinor/gordo-components/compare/0.4.0...0.5.0) (2019-01-11)

### Add

* Add Watchman service component

### Remove

* Remove manual env var parsing from Watchman


<a name="0.4.0"></a>
## [0.4.0](https://github.com/equinor/gordo-components/compare/0.3.1...0.4.0) (2019-01-08)

### Add

* Add unit tests

### Bump

* Bump pyyaml to 4.2b4 to fix vulnerability

### Clean

* Clean up tmp folder made by doctest

### Remove

* Remove machine_name in _datasets.py
* Remove s2i in Travis setup

### Save

* Save model metadata in model builder


<a name="0.3.1"></a>
## [0.3.1](https://github.com/equinor/gordo-components/compare/0.3.0...0.3.1) (2019-01-04)

### Base

* Base modelserver dockerimage on slim-stretch

### Bump

* Bump docker baseimage python:3.6.6->3.6.8

### Improve

* Improve docker caching in Dockerfile-ModelServer


<a name="0.3.0"></a>
## [0.3.0](https://github.com/equinor/gordo-components/compare/0.2.0...0.3.0) (2018-12-19)

### Add

* Add docker-based unit test for InfluxBackedDataset

### Minor

* Minor refactoring to _datasets.py

### Support

* Support owned gordo prediction server


<a name="0.2.0"></a>
## [0.2.0](https://github.com/equinor/gordo-components/compare/0.1.1...0.2.0) (2018-11-26)

### Convert

* Convert KerasAutoEncoder into transformer

### Fix

* Fix serializer from_pipeline definition mutation

### Set

* Set default log level to DEBUG

### Support

* Support FunctionTransformer steps in definition
* Support metadata saving with serializer.dumps


<a name="0.1.1"></a>
## [0.1.1](https://github.com/equinor/gordo-components/compare/0.1.0...0.1.1) (2018-11-15)

### Add

* Add TF graph & bump seldon serving image -> 0.3


<a name="0.1.0"></a>
## [0.1.0](https://github.com/equinor/gordo-components/compare/0.0.4...0.1.0) (2018-11-15)

### Add

* Add pinned requirements
* Add start/end dates in cli.py and _dataset.py
* Add mypy to testing
* Add doc tests to standard testing

### Build

* Build modelbuilder in docker

### Clean

* Clean the cli file, and add some TODOs to move away from environment variable overload

### Implement

* Implement auto versioning with SCM

### Print

* Print output from running cli in test

### Remove

* Remove tag_name in _datasets.py


<a name="0.0.4"></a>
## [0.0.4](https://github.com/equinor/gordo-components/compare/0.0.3...0.0.4) (2018-11-13)

### Add

* Add KerasAutoEncoder subclass of KerasModel
* Add pipeline deserialization from directory
* Add pipeline serialization to directory
* Add pipeline_into/from_definition functionality

### Integrate

* Integrate gordo_components.serializer with model builder

### Support

* Support updated MODEL_CONFIG with model build

### Update

* Update CLI model default to KerasAutoEncoder


<a name="0.0.3"></a>
## [0.0.3](https://github.com/equinor/gordo-components/compare/0.0.2...0.0.3) (2018-11-06)

### Add

* Add build_model_register decorator

### Update

* Update FeedForward AE to Scikit-Learn API


<a name="0.0.2"></a>
## [0.0.2](https://github.com/equinor/gordo-components/compare/0.0.1...0.0.2) (2018-10-30)

### Change

* Change mnt /data -> /gordo & fix tests

### Implement

* Implement InfluxBackedDataset


<a name="0.0.1"></a>
## 0.0.1 (2018-10-27)

### Add

* Add support for model saving to non-existant subdirs
* Add docs for each component

### Address

* Address Dataset API refactor

### Checkpoint

* Checkpoint - address review comments

### Convert

* Convert dockerfiles from setup.py to pip install .

### Fix

* Fix keras saving - file signature error
* Fix model build CMD & README update

### Ignore

* Ignore pickled objs

### Initial

* Initial commit
* Initial commit

### Integrate

* Integrate Health Indicator model

### Pushing

* Pushing docker images to azure

### Refactor

* Refactor Dataset API & implement autoencoder

### Rename

* Rename gordo-flow -> gordo-components

### Save

* Save Keras model via .h5 & others as .pkl

### Update

* Update Model server
* Update joblib loading of serialized model
* Update docs

