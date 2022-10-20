import os
from pathlib import Path
import logging
import pytest
import churn_library as churnlib

test_log = logging.getLogger(__name__)

backup_path = Path(__file__).parent / "backup"
backup_path.mkdir(parents=True, exist_ok=True)


class Vars:
    bank_data_file = "./data/bank_data.csv"


def log_assert(expression, msg):
    """
    Function which asserts given expression and
    logs provided message if assertion fails
    """
    if expression:
        return
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
            return func(*args, **kwargs)
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
        func=churnlib.ChurnLibrary,
        error=FileNotFoundError,
        error_msg="Testing import_eda: The file wasn't found",
    )(Vars.bank_data_file).df

    test_log.info("Testing import_data: SUCCESS")

    log_assert(
        df.shape[0] > 0,
        "Testing import_data: The file doesn't appear to have rows and columns")
    log_assert(
        df.shape[1] > 0,
        "Testing import_data: The file doesn't appear to have rows and columns")


def test_eda(churn_library_object):
    '''
        test perform eda function
        '''
    # Delete existing image files
    image_names = churnlib.consts.eda_plots.values()
    for fp in churnlib.images_path.iterdir():
        if fp.name in image_names:
            fp.rename(backup_path / f"backup_{fp.name}")

    churn_library_object.perform_eda()
    image_path_list = [
        fp.name for fp in churnlib.images_path.iterdir() if fp.is_file()]
    for fname in image_names:
        log_assert(fname in image_path_list, f"{fname} not generated in EDA")

    test_log.info("Testing EDA: SUCCESS")


def test_encoder_helper(churn_library_object):
    '''
        test encoder helper
        '''
    churn_library_object.encoder_helper_loop()
    for cat_col, cat_col_new in churnlib.consts.cat_columns.items():
        log_assert(cat_col_new in churn_library_object.df.columns, f"No encoding done for {cat_col}")

    test_log.info("Testing Encoder: SUCCESS")


def test_perform_feature_engineering(churn_library_object):
    '''
        test perform_feature_engineering
        '''
    churn_library_object.perform_feature_engineering()
    log_assert(set(churn_library_object.cl_vars.X.columns.values) == set(
        churnlib.consts.quant_columns), "Issue in selection of numerical columns")
    test_log.info("Testing Feature Engineering: SUCCESS")


def test_train_models(churn_library_object):
    '''
        test train_models
        '''
    # Backup existing models
    for key, val in churnlib.consts.model_names.items():
        Path(churnlib.models_path / val).rename(backup_path / f"backup_{val}")
    churn_library_object.train_models()
    models_path_list = [
        fp.name for fp in churnlib.models_path.iterdir() if fp.is_file()]
    for key, val in churnlib.consts.model_names.items():
        log_assert(val in models_path_list, f"{val} model not generated during training")

    test_log.info("Testing Model training: SUCCESS")


def test_results(churn_library_object):
    for key, val in churnlib.consts.result_plots.items():
        Path(churnlib.images_path / val).rename(backup_path / f"backup_{val}")

    churn_library_object.roc_plot()
    churn_library_object.feature_importance_plot()
    churn_library_object.classification_report_image()

    results_path_list = [
        fp.name for fp in churnlib.images_path.iterdir() if fp.is_file()]
    for key, val in churnlib.consts.result_plots.items():
        log_assert(val in results_path_list, f"{val} not generated after training")

    test_log.info("Testing Report Generation: SUCCESS")


if __name__ == "__main__":
    pass
