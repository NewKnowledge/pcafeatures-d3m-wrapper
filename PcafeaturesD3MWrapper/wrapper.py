import os.path
import numpy as np
import pandas
import pickle
import requests
import ast
import typing
from json import JSONDecoder
from typing import List

from punk.feature_selection import PCAFeatures
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base, params

__author__ = 'Distil'
__version__ = '3.0.0'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass


class Hyperparams(hyperparams.Hyperparams):
    pass

class pcafeatures(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "04573880-d64f-4791-8932-52b7c3877639",
        'version': __version__,
        'name': "PCA Features",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Rank and score numeric features based on principal component analysis'],
        'source': {
            'name': __author__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/pcafeatures-d3m-wrapper",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
         'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/pcafeatures-d3m-wrapper.git@{git_commit}#egg=PcafeaturesD3MWrapper'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.distil.pcafeatures',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,

    })
    
    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
                
        self._decoder = JSONDecoder()
        self._params = {}

    def fit(self) -> None:
        pass
    
    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        pass
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Perform principal component analysis on all numeric data in the dataset
        and then use each original features contribution to the first principal
        component as a proxy for the 'score' of that feature. Returns a dataframe
        containing an ordered list of all original features as well as their 
        contribution to the first principal component.
        Parameters
        ----------
        inputs : Input pandas frame

        Returns
        -------
        Outputs : pandas frame with list of original features in first column, ordered
            by their contribution to the first principal component, and scores in
            the second column.
        """
        return CallResult(PCAFeatures().rank_features(inputs = inputs))


if __name__ == '__main__':
    client = pcafeatures(hyperparams={})
    # make sure to read dataframe as string!
    # frame = pandas.read_csv("https://query.data.world/s/10k6mmjmeeu0xlw5vt6ajry05",dtype='str')
    frame = pandas.read_csv("https://s3.amazonaws.com/d3m-data/merged_o_data/o_4550_merged.csv",dtype='str')
    result = client.produce(inputs = frame)
    print(result)
