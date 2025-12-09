import os
import datasets
import pandas as pd

class watchesConfig(datasets.BuilderConfig):
    def __init__(self, features, data_url, **kwargs):
        super(watchesConfig, self).__init__(**kwargs)
        self.features = features
        self.data_url = data_url

class watches(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        watchesConfig(
            name="pairs",
            features={
                "ltable_id":datasets.Value("string"),
                "rtable_id":datasets.Value("string"),
                "label":datasets.Value("string"),
            },
            data_url="https://huggingface.co/datasets/matchbench/WDC-watches/resolve/main/",
        ),
        watchesConfig(
            name="source",
            features={
                "id":datasets.Value("string"),
                "title":datasets.Value("string"),
            },
            data_url="https://huggingface.co/datasets/matchbench/WDC-watches/resolve/main/tableA.csv",
        ),
        watchesConfig(
            name="target",
            features={
                "id":datasets.Value("string"),
                "title":datasets.Value("string"),
            },
            data_url="https://huggingface.co/datasets/matchbench/WDC-watches/resolve/main/tableB.csv",
        ),
    ]
    
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(self.config.features)
        )
    
    def _split_generators(self, dl_manager):
        if self.config.name == "pairs":
            return [
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "path_file": dl_manager.download_and_extract(os.path.join(self.config.data_url, f"{split}.csv")),
                        "split":split,
                    }
                )
                for split in ["train", "valid", "test"]
            ]
        if self.config.name == "source":
            return [ datasets.SplitGenerator(name="source",gen_kwargs={"path_file":dl_manager.download_and_extract(self.config.data_url), "split":"source",})]
        if self.config.name == "target":
            return [ datasets.SplitGenerator(name="target",gen_kwargs={"path_file":dl_manager.download_and_extract(self.config.data_url), "split":"target",})]
    
    
    
    def _generate_examples(self, path_file, split):
        file = pd.read_csv(path_file)
        for i, row in file.iterrows():
            if split not in ['source', 'target']:
                yield i, {
                    "ltable_id": row["ltable_id"],
                    "rtable_id": row["rtable_id"],
                    "label": row["label"],
                }
            else:
                yield i, {
                    "id": row["id"],
                    "title": row["title"],
                }