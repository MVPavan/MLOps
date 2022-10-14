import os
import logging
import pytest
import churn_library as churnlib

test_log = logging.getLogger(__name__)

class Vars:
    bank_data_file = "./data/bank_data.csv"


def log_assert(expression,msg):
    """
    Function which asserts given expression and
    logs provided message if assertion fails
    """
    if expression:return
    test_log.error(msg)
    assert expression, msg

def log_try_except(func, error, error_msg):
    """
    executes func in try and except block
    if rasied error matches provide error logs the error msg
    usage: log_try_except(function, ErrorType, "error message")(function arguments) 
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args,**kwargs)
        except error as err:
            test_log.error(error_msg)
            raise err
    return wrapper


@pytest.fixture(scope="session")
def churn_library_object():
    """
    Returns dataframe
    """
    return churnlib.ChurnLibrary(Vars.bank_data_file)


def test_import():
    """
	test data import - this example is completed for you to assist with the other test functions
	"""
    df = log_try_except(
        func = churnlib.import_data,
        error=FileNotFoundError,
        error_msg="Testing import_eda: The file wasn't found",
    )(Vars.bank_data_file)

    test_log.info("Testing import_data: SUCCESS")

    log_assert(df.shape[0] > 0, "Testing import_data: The file doesn't appear to have rows and columns")
    log_assert(df.shape[1] > 0, "Testing import_data: The file doesn't appear to have rows and columns")


def test_eda(import_bank_data):
    '''
	test perform eda function
	'''
    df = import_bank_data
    # Delete existing image files
    image_names = churnlib.consts.eda_plots.values()
    for fp in churnlib.images_path.iterdir():
        if fp.name in image_names: fp.unlink()

    churnlib.perform_eda(df)
    image_path_list = [fp.name for fp in churnlib.images_path.iterdir() if fp.is_file()]
    for fname in image_names:
        log_assert(fname in image_path_list,f"{fname} not generated in EDA")
    


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
