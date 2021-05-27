# Repository for the master's thesis: *A Deep Study of Deep Hedging*
The structure of the repository is as follows:
* **experiments** contains the scripts for running the various experiments in the thesis.
* **noncode** contains the Excel-files encompassing all results from the aforementioned experiments.
* **results** contains raw files from the aforementioned experiments.
* **tests** contains unittests. 
* approximators.py contains the approximator objects such as neural networks.
* books.py defines the derivative book class.
* constants.py defines constant variables.
* derivatives.py contains class definitions of various derivatives to be used in the derivative book.
* gradient_driver.py contains a driver for training and testing teh gradient models.
* gradient_models.py implements the models using adjoints.
* hedge_models.py implements the various trainable hedging models.
* preprocessing.py implements preprocessors such as normalisation, PCA etc.
* random_books.py contains functions for generating random derivative books.
* simulators.py contains market simulators.
* utils.py contains various helper functions and importantly the HedgeDriver class which aids in training and testing of hedge models.
