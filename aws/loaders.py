import collections
import csv
import io
import json
from .s3 import S3



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


