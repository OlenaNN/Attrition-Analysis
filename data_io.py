import csv
import logging
from time import gmtime, strftime

from aws import loaders, savers
from aws import s3

logger = logging.getLogger(__name__)


class ModelArtifactsIO:

    def __init__(self, s3_bucket, labeled_data_folder=''):
        self.s3_bucket = s3_bucket
        self.model_name = 'model_name'
        self.labeled_data_folder = labeled_data_folder
        self.base_labeled_data_folder = 'labeled_data'
        self.artifacts_folder = 'model_artifacts'
        self.model_adjuster_config_name = 'model_adjuster_config.json'
        self.test_dataset_name = 'test_dataset.csv'
        self.train_dataset_name = 'train_dataset.csv'
        self.resultset_name = 'ma_resultset.json'
        self.data_savers = {
            'csv': savers.S3CSVSaver(),
            'rnn_model': savers.S3RnnModelSaver(),
            'json': savers.S3JSONSaver()
        }
        self.data_loaders = {
            'csv': loaders.S3CSVLoader(),
            'json': loaders.S3JSONLoader()
        }

    def build_model_path(self, models_version, model_name):
        return f'{self.artifacts_folder}/{models_version}}'

    def build_model_metrics_path(self, models_version, model_name):
        return f'{self.build_model_path(models_version, model_name)}/metrics.json'

    def build_confusion_matrix_path(self, models_version, model_name):
        return f'{self.build_model_path(models_version, model_name)}/confusion_matrix.json'

    def build_domain_config_path(self, models_version):
        return f'{self.domain_name}/{self.artifacts_folder}/{models_version}/{self.model_adjuster_config_name}'

    def build_qa_dataset_path(self, models_version):
        return f'{self.artifacts_folder}/{models_version}/{self.test_dataset_name}'

    def build_train_dataset_path(self, models_version):
        return f'{self.artifacts_folder}/{models_version}/{self.train_dataset_name}'

    def build_invalid_dataset_path(self, models_version):
        return f'{self.domain_name}/{self.artifacts_folder}/{models_version}/{self.invalid_dataset_name}'

    def build_train_categorical_dataset_path(self, models_version, model_name):
        return f'{self.domain_name}/{self.artifacts_folder}/{models_version}/{model_name}/{self.train_categorical_name}'

    def build_qa_categorical_dataset_path(self, models_version, model_name):
        return f'{self.domain_name}/{self.artifacts_folder}/{models_version}/{model_name}/{self.qa_categorical_name}'

    def build_invalid_categorical_dataset_path(self, models_version, model_name):
        return f'{self.domain_name}/{self.artifacts_folder}/{models_version}/{model_name}/{self.invalid_categorival}'

    def build_resultset_path(self, models_version):
        return f'{self.artifacts_folder}/{models_version}/{self.resultset_name}'

    def build_validation_metrics_path(self, models_version):
        return f'{self.domain_name}/{self.artifacts_folder}/{models_version}/{self.validation_metrics}'

    def build_model_adjuster_config_path(self, domain_name):
        return f'{domain_name}/{self.model_adjuster_config_name}'

    def build_labeled_data_input_path(self):
        return f'{self.base_labeled_data_folder}/{self.labeled_data_folder}'
    ###
    def build_model_path(self, models_version, model_name):
        return f'{self.model_name}/{self.artifacts_folder}/{models_version}/{model_name}'


    def build_test_dataset_path(self, models_version):
        return f'{self.model_name}/{self.artifacts_folder}/{models_version}/{self.test_dataset_name}'

    def build_train_dataset_path(self, models_version):
        return f'{self.model_name}/{self.artifacts_folder}/{models_version}/{self.train_dataset_name}'


    def build_resultset_path(self, models_version):
        return f'{self.model_name}/{self.artifacts_folder}/{models_version}/{self.resultset_name}'


    def build_model_adjuster_config_path(self, domain_name):
        return f'{domain_name}/{self.model_adjuster_config_name}'

    def get_ma_config(self):
        key = self.build_model_adjuster_config_path(self.model_name)
        ma_config = self.load_json(key)
        return ma_config

    def s3_file_exists(self, path):
        s = s3.S3()
        for file in s.get_s3_objects(self.s3_bucket, path, True):
            if file.key == path:
                return True
        return False

    def get_all_csv_files(self, path):
        s = s3.S3()
        csv_list = []
        for file in s.get_s3_objects(self.s3_bucket, path, True):
            if '.csv' in file.key:
                csv_list.append(file)
        return csv_list

    def save_csv(self, path, data):
        self.data_savers['csv'].save(self.s3_bucket, path, data)

    def load_csv(self, path):
        return self.data_loaders['csv'].load(self.s3_bucket, path)

    def save_json(self, path, data):
        self.data_savers['json'].save(self.s3_bucket, path, data)

    def load_json(self, path):
        return self.data_loaders['json'].load(self.s3_bucket, path)

    def save_model(self, path, model):
        self.data_savers['model'].save(self.s3_bucket, path, model)

    @staticmethod
    def build_models_version():
        return strftime("%Y%m%d%H%M", gmtime())


def save_dict_to_csv(dict, filename):
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=dict[0].keys())
        writer.writeheader()
        for record in dict:
            writer.writerow(record)

