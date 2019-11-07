import collections
import csv
import io
import json
import os
import tempfile

from .s3 import S3
from .sql import SQL


class LoaderBase:

    def load(self, *args, **kwargs):
        raise NotImplementedError


class S3CSVLoader(S3, LoaderBase):

    def load(self, bucket, key):
        """
        Downloads a CSV file from S3 and then iterates over its parsed lines.
        :param bucket: S3 bucket name
        :param key: S3 key name (i.e. CSV file path)
        :return: iterable of OrderedDict objects
        """

        bucket = self._build_bucket_resource(bucket)

        with io.BytesIO() as stream:
            bucket.download_fileobj(key, stream)
            stream.seek(0)

            wrapper = io.TextIOWrapper(stream, encoding='utf-8')
            reader = csv.DictReader(wrapper)
            yield from reader


class S3JSONLoader(S3, LoaderBase):

    def load(self, bucket, key):
        """
        Downloads a JSON file from S3 and then deserializes data from it.
        :param bucket: S3 bucket name
        :param key: S3 key name (i.e. JSON file path)
        :return: deserialized JSON object
        """

        bucket = self._build_bucket_resource(bucket)

        with io.BytesIO() as stream:
            bucket.download_fileobj(key, stream)
            stream.seek(0)

            wrapper = io.TextIOWrapper(stream, encoding='utf-8')
            # Preserve the original order
            return json.load(wrapper, object_pairs_hook=collections.OrderedDict)


class SQLLoader(SQL, LoaderBase):

    def load(self, host, port, dbname, user, password, query):
        """
        Executes some SELECT query and then iterates over its fetched records.
        :param host, port, dbname, user, password: DB connection parameters
        :param query: DB SQL query string
        :return: iterable of OrderedDict objects
        """

        with self(host, port, dbname, user, password) as cursor:
            cursor.execute(query)
            fieldnames = tuple(field.name for field in cursor.description)
            for record in cursor:
                yield collections.OrderedDict(zip(fieldnames, record))


class S3RnnModelLoader(S3, LoaderBase):

    def load(self, bucket, key, rnn_class):
        """
        Loads the specified RNN model from S3.
        :param bucket: S3 bucket name
        :param key: S3 key name (i.e. RNN data folder path)
        :param rnn_class: RNN class
        :return: RNN object
        """

        bucket = self._build_bucket_resource(bucket)
        dir_path = key  # alias

        with tempfile.TemporaryDirectory() as tempdir_name:
            for key in self.get_s3_keys(bucket=bucket.name, prefix=dir_path, only_files=True):
                file_name = key.rsplit('/', 1)[-1]
                local_file_path = os.path.join(tempdir_name, file_name)
                bucket.download_file(key, local_file_path)

            return rnn_class(load_path=tempdir_name)
