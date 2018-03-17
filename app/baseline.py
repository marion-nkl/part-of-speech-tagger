from app.data_fetcher import DataFetcher
import pandas as pd
from collections import Counter


def create_baseline_mapper():
    """

    :return:
    """
    data_dict = DataFetcher.read_data(files_list=['train', 'dev'])

    train_data = DataFetcher.parse_conllu(data_dict['train'])
    dev_data = DataFetcher.parse_conllu(data_dict['dev'])

    out = list()
    for sentence in train_data + dev_data:
        if sentence:
            for t in sentence:
                x = t[0].strip().lower()
                out.append((x, t[1]))

    df = pd.DataFrame(out, columns=['word', 'pos_tag'])

    out_dict = dict()
    for w, group in df.groupby(['word']):
        out_dict[w] = Counter(group['pos_tag']).most_common(1)[0][0]

    return out_dict
