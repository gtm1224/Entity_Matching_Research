import os
import datasets
import pandas as pd

class ItunesAmazonConfig(datasets.BuilderConfig):
    def __init__(self, features, data_url, **kwargs):
        super(ItunesAmazonConfig, self).__init__(**kwargs)
        self.features = features
        self.data_url = data_url

class ItunesAmazon(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ItunesAmazonConfig(
            name="pairs",
            features={
                "ltable_id":datasets.Value("string"),
                "rtable_id":datasets.Value("string"),
                "label":datasets.Value("string"),
            },
            data_url="https://huggingface.co/datasets/matchbench/Itunes-Amazon/resolve/main/",
        ),
        ItunesAmazonConfig(
            name="source",
            features={
                "id":datasets.Value("string"),
                "Song_Name":datasets.Value("string"),
                "Artist_Name":datasets.Value("string"),
                "Album_Name":datasets.Value("string"),
				"Genre":datasets.Value("string"),
				"Price":datasets.Value("string"),
				"CopyRight":datasets.Value("string"),
				"Time":datasets.Value("string"),
				"Released":datasets.Value("string"),
            },
            data_url="https://huggingface.co/datasets/matchbench/Itunes-Amazon/resolve/main/tableA.csv",
        ),
        ItunesAmazonConfig(
            name="target",
            features={
                "id":datasets.Value("string"),
                "Song_Name":datasets.Value("string"),
                "Artist_Name":datasets.Value("string"),
                "Album_Name":datasets.Value("string"),
				"Genre":datasets.Value("string"),
				"Price":datasets.Value("string"),
				"CopyRight":datasets.Value("string"),
				"Time":datasets.Value("string"),
				"Released":datasets.Value("string"),
            },
            data_url="https://huggingface.co/datasets/matchbench/Itunes-Amazon/resolve/main/tableB.csv",
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
                    "Song_Name": row["Song_Name"],
                    "Artist_Name": row["Artist_Name"],
                    "Album_Name": row["Album_Name"],
					"Genre": row["Genre"],
					"Price": row["Price"],
					"CopyRight": row["CopyRight"],
					"Time": row["Time"],
					"Released": row["Released"],
                }