import functools
import logging
import data_processing
from model_utils import LGBMModel
from hyperopt import tpe, fmin
from data_io import ModelArtifactsIO
#from decorator import log_time

logger = logging.getLogger(__name__)


def notify_on_exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            # The actual error traceback will be logged automatically
            logger.exception('Unexpected error occurred')
            raise
    return wrapper


class ModelAdjuster:
    def __init__(self, env_vars):
        logger.debug(f'Initializing ModelAdjuster from Config')
        self.ma_artifacts_io = ModelArtifactsIO(env_vars.S3_BUCKET, env_vars.LABELED_DATA_FOLDER)
        self.labeled_data_folder = env_vars.LABELED_DATA_FOLDER

    @notify_on_exception
    def run(self):

        # Retrieve Labeled Data
        labeled_data = self.get_labeled_data()

        preprocessed_data = self.preprocess_labeled_data(labeled_data)

        final_lgbm_model, parameters, train_x, test_x, models_version = self.train_model(preprocessed_data)

        # Publish Model Statistics
        self.write_model_artifacts(final_lgbm_model, parameters, train_x, test_x, models_version)

    def get_labeled_data(self):
        """
        get all csv files specified in labeled_data directory
        return them in the same format as BE API would
        @param domain: the domain config object
        @return: list of dicts (rows in csv file(s) )
        """  # track how long this takes
        logger.debug(f'S3 path: {self.labeled_data_folder}')

        data_path = self.ma_artifacts_io.build_labeled_data_input_path()
        logger.info(f'Getting S3 data from {data_path}')

        csv_list = self.ma_artifacts_io.get_all_csv_files(data_path)
        logger.info(csv_list)

        deserialized_data = []
        for csv in csv_list:
            if not self.ma_artifacts_io.s3_file_exists(csv.key):
                raise FileNotFoundError(f'File not found in S3 at {csv.key}')
            logger.info(f'reading from file {csv.key}')
            data = self.ma_artifacts_io.load_csv(csv.key)
            #DESERIALIZE LABELED DATA (DOMAIN META)
            data = [x for x in data]
            logger.info(f'adding {len(data)} records to dataset')
            deserialized_data += data

        logger.info(f'retrieved {len(deserialized_data)} records')
        return deserialized_data

#    @log_time(logger)
    def preprocess_labeled_data(self, data):
        numerical_data, categorical_data, data_description = data_processing.generate_data_description(data)
        data_processed = data_processing.data_preparation(data, categorical_data)
        return data_processed


#    @log_time(logger)
    def train_model(self, data):
        lgbm_model = LGBMModel()
        train_x, test_x, train_y, test_y  = lgbm_model.split_data(data)
        lgb_hyperparams = fmin(fn=lgbm_model.lgb_objective,
                           max_evals=150,
                           trials=lgbm_model.trials,
                           algo=tpe.suggest,
                           space=lgbm_model.space
                           )
        lgb_results = lgbm_model.generate_model_report(lgb_hyperparams)
        parameters = lgb_results['parameters']
        final_lgbm_model = lgbm_model.fit_final_model(parameters)
        models_version = ModelArtifactsIO.build_models_version()
        return final_lgbm_model, lgb_results, train_x, test_x, models_version



#    @log_time(logger)
    def write_model_artifacts(self, final_lgbm_model, parameters, train_x, test_x, models_version):
        # write back to S3
        logger.info('write model artifacts')

        # Save final results only if all previous steps passed successfully
        logger.info(f'Building models for version: {models_version}')

        logger.info('Saving sampled QA dataset to S3')
        qa_dataset_path = self.ma_artifacts_io.build_qa_dataset_path(
            models_version)

        self.ma_artifacts_io.save_csv(qa_dataset_path, test_x)

        logger.info('Saving sampled TRAIN dataset to S3')
        train_dataset_path = self.ma_artifacts_io.build_train_dataset_path(
            models_version)
        self.ma_artifacts_io.save_csv(
            train_dataset_path, train_x)


        logger.info('Saving final MA resultset to S3')
        resultset_path = self.ma_artifacts_io.build_resultset_path(
            models_version)
        self.ma_artifacts_io.save_json(resultset_path, parameters)

        logger.info(f'Uploading final model to S3')

        model_path = self.ma_artifacts_io.build_model_path(models_version)
        self.model_json = final_lgbm_model.booster_.dump_model()
        self.ma_artifacts_io.save_json(model_path, self.model_json )


