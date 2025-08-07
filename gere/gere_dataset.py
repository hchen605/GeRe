# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import json
import os
import datasets
# import ipdb

logger = datasets.logging.get_logger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

def save_ds(instances, file_name):
    with open(file_name, "w+", encoding='utf-8') as fi:
        json.dump(instances, fi, ensure_ascii=False, indent=2)


def load_jsonl(jsonl_file):
    """read data by line with json"""
    with open(jsonl_file, 'r', encoding='U8') as f:
        return [json.loads(line.strip()) for line in f]


class GeReConfig(datasets.BuilderConfig):
    def __init__(
            self,
            *args,
            data_args=None,
            data_file='./slim_redpajama/slim_redpajama.sampled_of_1_chunk.head1k.jsonl',
            gere_dataset_name='SlimRedpajama',
            num_gere_samples=1000,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_args = data_args
        self.data_file = data_file
        self.gere_dataset_name = gere_dataset_name
        self.num_gere_samples = num_gere_samples


class GeReDataset(datasets.GeneratorBasedBuilder):
    """General Replay Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = GeReConfig
    BUILDER_CONFIGS = []
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "GeRe": {
                        "task": datasets.Value("string"),
                        "dataset": datasets.Value("string"),
                        "id": datasets.Value("string"),
                        "text": datasets.Value("string"),
                    }
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_file is None:
            logger.error("Please provide right input: data_file!")


        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": self.config.data_file,
                    "num_gere_samples": self.config.num_gere_samples,
                    "gere_dataset_name": self.config.gere_dataset_name,
                }),
        ]

    def load_SlimRedpajama_dataset(self, data_file, num_samples=1000):
        instances = load_jsonl(data_file)
        if num_samples is not None:
            instances = instances[:num_samples]

        for idx, instance in enumerate(instances):
            sample = {
                'GeRe':
                    {
                        'task': 'GeRe',
                        'dataset': 'SlimRedpajama',
                        'id': str(idx),
                        'text': instance['text'],
                    }
            }
            yield sample

    def _generate_examples(self, data_file, num_gere_samples=None, gere_dataset_name=None):
        logger.info(f"num_gere_samples: {num_gere_samples}")

        if gere_dataset_name is None or gere_dataset_name=='SlimRedpajama':
            for sample in self.load_SlimRedpajama_dataset(
                    data_file,
                    num_samples=num_gere_samples
            ):
                GeRe = sample['GeRe']
                yield f"{GeRe['task']}#{GeRe['dataset']}#{GeRe['id']}", sample


if __name__ == "__main__":
    pass