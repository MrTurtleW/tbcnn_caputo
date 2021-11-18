import os
import re
import sys
import json
import shutil
import hashlib

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '..'))

from config import black_sample_path, white_sample_path, sample_meta_path, black_output_path, white_output_path

file_size_threshold_max = 50000
file_size_threshold_min = 200


def get_sha256(data: str):
    sha256 = hashlib.sha256()
    sha256.update(data.encode())
    return sha256.hexdigest()


class JavaScriptExtractor:
    def __init__(self):
        with open(sample_meta_path, 'r') as f:
            data = f.read()

        self.meta = json.loads(data)

    def main(self):
        self.process(black_sample_path, black_output_path)
        self.process(white_sample_path, white_output_path)

    def process(self, path, output_path):
        print('processing {}'.format(path))
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.makedirs(output_path)
        for filename in os.listdir(path):
            source = os.path.join(path, filename)
            if self.meta[filename]['file_type'] == 'CDE_FT_HTML':
                javascript = self.get_javascript_from_html(source)
                if not javascript:
                    continue

                sha256 = get_sha256(javascript)
                target = os.path.join(output_path, sha256)
                with open(target, 'w', encoding='utf-8') as f:
                    f.write(javascript)
            else:
                target = os.path.join(output_path, filename)
                shutil.copy(os.path.join(path, filename), target)

            size = os.path.getsize(target)
            if file_size_threshold_min < size < file_size_threshold_max:
                pass
            else:
                os.remove(target)

    def get_javascript_from_html(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = f.read()
        except UnicodeDecodeError:
            return None

        result = []
        for code in re.findall(r'<script>(.*?)</script>', data):
            if code:
                result.append(code)

        return "\n".join(result)


if __name__ == '__main__':
    extractor = JavaScriptExtractor()
    extractor.main()
