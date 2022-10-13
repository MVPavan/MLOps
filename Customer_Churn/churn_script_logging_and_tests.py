import os
import logging
import pytest
import churn_library as churnlib

# logging.basicConfig(
#     filename='./logs/churn_library.log',
#     level=logging.INFO,
#     filemode='w+',
#     format='%(name)s - %(levelname)s - %(message)s')
# test_log = logging.getLogger()
test_log = logging.getLogger(__name__)

class Vars:
    bank_data_file = "./data/bank_data.csv"

@pytest.fixture(scope="session")
def import_bank_data():
    """
    Returns dataframe
    """
    test_log.info("Opening file")
    return churnlib.import_data(Vars.bank_data_file)


def test_import():
    """
	test data import - this example is completed for you to assist with the other test functions
	"""
    try:
        df = churnlib.import_data(Vars.bank_data_file)
        test_log.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        test_log.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        test_log.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(import_bank_data):
    '''
	test perform eda function
	'''
    df = import_bank_data
    churnlib.perform_eda(df)
    image_path_list = [fp for fp in churnlib.images_path.iterdir() if fp.is_file()]
    


# def test_encoder_helper(encoder_helper):
#     '''
# 	test encoder helper
# 	'''


# def test_perform_feature_engineering(perform_feature_engineering):
#     '''
# 	test perform_feature_engineering
# 	'''


# def test_train_models(train_models):
#     '''
# 	test train_models
# 	'''


if __name__ == "__main__":
    pass
