import csv
import io
import json
from .s3 import S3
import tempfile
import os
import gzip

class SaverBase:

    def save(self, *args, **kwargs):
        raise NotImplementedError


class S3CSVSaver(S3, SaverBase):

    def save(self, bucket, key, entries, compress=False):
        """
        Writes data into a CSV file and then uploads it to S3.
        :param bucket: S3 bucket name
        :param key: S3 key name (i.e. CSV file path)
        :param entries: iterable of dict-like objects
        :param compress: boolean, default False, that toggles gzip compression of data files
        :return: None
        """

        entries_iterator = iter(entries)

        bucket = self._build_bucket_resource(bucket)

        with io.BytesIO() as stream:
            if compress:
                wrapper = io.TextIOWrapper(gzip.GzipFile(fileobj=stream, mode='wb'), encoding='utf-8', newline='\n')
                # append extension to key (which ends in <file_name>.csv)
                key = key + '.gz'
            else:
                wrapper = io.TextIOWrapper(stream, encoding='utf-8', newline='\n')

            # Try to fetch the very first entry in order to
            # extract the actual field names (required for DictWriter)
            try:
                first_entry = next(entries_iterator)
            except StopIteration:
                return
            else:
                fieldnames = tuple(first_entry.keys())

            writer = csv.DictWriter(
                wrapper, fieldnames=fieldnames, lineterminator='\n'
            )
            writer.writeheader()
            writer.writerow(first_entry)
            writer.writerows(entries_iterator)

            if compress:
                # Make sure to close the compressed buffer
                wrapper.close()
            else:
                # Make sure to flush the buffer in order to
                # write everything to the underlying stream
                wrapper.flush()

            stream.seek(0)
            bucket.upload_fileobj(stream, key)


class S3JSONSaver(S3, SaverBase):

    def save(self, bucket, key, data, encoder=None):
        """
        Serializes data into a JSON file and then uploads it to S3.
        :param bucket: S3 bucket name
        :param key: S3 key name (i.e. JSON file path)
        :param data: serializable JSON object
        :param encoder: optional custom json.JSONEncoder
        :return: None
        """

        bucket = self._build_bucket_resource(bucket)

        with io.BytesIO() as stream:
            wrapper = io.TextIOWrapper(stream, encoding='utf-8', newline='\n')

            json.dump(data, wrapper, indent=2, cls=encoder)

            # Make sure to flush the buffer in order to
            # write everything to the underlying stream
            wrapper.flush()

            stream.seek(0)
            bucket.upload_fileobj(stream, key)




