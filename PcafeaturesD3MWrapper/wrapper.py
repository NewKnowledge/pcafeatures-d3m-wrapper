import os.path
import pandas

from punk.feature_selection import PCAFeatures
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame

__author__ = 'Distil'
__version__ = '3.0.1'
__contact__ = 'jeffrey.gleason@newknowledge.io'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    pass

class pcafeatures(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
        Perform principal component analysis on all numeric data in the dataset
        and then use each original features contribution to the first principal
        component as a proxy for the 'score' of that feature. Returns a dataframe
        containing an ordered list of all original features as well as their 
        contribution to the first principal component.
    """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "04573880-d64f-4791-8932-52b7c3877639",
        'version': __version__,
        'name': "PCA Features",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Rank and score numeric features based on principal component analysis'],
        'source': {
            'name': __author__,
            'contact': __contact__,
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
        'python_path': 'd3m.primitives.feature_selection.pca_features.Pcafeatures',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,

    })
    
    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Parameters
        -------
        inputs : Input pandas frame

        Returns
        -------
        Outputs : pandas frame with list of original features in first column, ordered
            by their contribution to the first principal component, and scores in
            the second column.
        """

        # add metadata to output data frame
        pca_df = d3m_DataFrame(PCAFeatures().rank_features(inputs = inputs))
        # first column ('features')
        col_dict = dict(pca_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'features'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        pca_df.metadata = pca_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        # second column ('scores')
        col_dict = dict(pca_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1.0")
        col_dict['name'] = 'scores'
        col_dict['semantic_types'] = ('http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        pca_df.metadata = pca_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)

        return CallResult(pca_df)


if __name__ == '__main__':
    # LOAD DATA AND PREPROCESSING
    input_dataset = container.Dataset.load('file:///data/home/jgleason/D3m/datasets/seed_datasets_current/196_autoMpg/196_autoMpg_dataset/datasetDoc.json') 
    ds2df_client = DatasetToDataFrame(hyperparams={"dataframe_resource":"0"})
    df = ds2df_client.produce(inputs = input_dataset)   
    client = pcafeatures(hyperparams={})
    # make sure to read dataframe as string!
    # frame = pandas.read_csv("https://query.data.world/s/10k6mmjmeeu0xlw5vt6ajry05",dtype='str')
    #frame = pandas.read_csv("https://s3.amazonaws.com/d3m-data/merged_o_data/o_4550_merged.csv",dtype='str')
    result = client.produce(inputs = df.value)
    bestFeatures = [int(row[1]) for row in result.value.itertuples() if float(row[2]) > 0.1]
    print(bestFeatures)
