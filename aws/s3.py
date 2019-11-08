import os

try:
    import boto3
except ImportError:
    import warnings
    warnings.warn('Cannot import boto3, S3 will not work.')


class S3:

    def __init__(self):

        options = {
            'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
            'region_name': os.environ.get('REGION'),
        }
        self.s3 = boto3.resource('s3', **options)

    def _build_bucket_resource(self, bucket):
        return self.s3.Bucket(bucket)

    def get_s3_objects(self, bucket, prefix='', only_files=False, only_folders=False):
        """
        Collects all objects from the S3 bucket.

        :param bucket: S3 bucket name
        :param prefix: fetches only keys which begin with the specified prefix
        :param only_files: fetches only files (skipping "folders") if True
        :param only_folders: fetches only folders (skipping "files") if True
        :return: iterable of s3 objects
        """

        bucket = self._build_bucket_resource(bucket)
        for obj in bucket.objects.filter(Prefix=prefix):
            if only_files and obj.key.endswith('/'):
                continue
            if only_folders and not obj.key.endswith('/'):
                continue
            yield obj

    def get_s3_keys(self, bucket, prefix='', only_files=False, only_folders=False):
        """
        Collects all keys from the S3 bucket.

        :param bucket: S3 bucket name
        :param prefix: fetches only keys which begin with the specified prefix
        :param only_files: fetches only files (skipping "folders") if True
        :param only_folders: fetches only folders (skipping "files") if True
        :return: iterable of str objects
        """

        for obj in self.get_s3_objects(bucket, prefix=prefix, only_files=only_files, only_folders=only_folders):
            yield obj.key

    def delete_s3_keys(self, bucket, keys):
        """
        Deletes the specified keys from the S3 bucket.
        Does nothing for non-existing keys (skipping them silently).

        :param bucket: S3 bucket name
        :param keys: iterable of S3 keys names
        :return: None
        """

        bucket = self._build_bucket_resource(bucket)

        # AWS API allows to delete up to 1000 keys per each HTTP request

        objects = []
        for key in keys:
            objects.append({'Key': key})
            if len(objects) == 1000:
                bucket.delete_objects(Delete={'Objects': objects})
                objects = []

        if objects:
            bucket.delete_objects(Delete={'Objects': objects})
