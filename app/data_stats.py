import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from app.data_fetcher import DataFetcher


def plot_data_stats(data):
    """
    Plots the data distribution
    :param data: dataframe
    :return: plot
    """
    sns.set_style("dark")
    f, ax = plt.subplots(figsize=(6, 15))

    ax = sns.barplot(x='tag', y='count', data=tags_freqs)

    ax.axes.set_title("POS Tags Frequencies",fontsize=20)
    ax.set_xlabel("POS Tags", fontsize=16)
    ax.set_ylabel("Counts", fontsize=16)
    ax.tick_params(labelsize=12)

    plt.show()


if __name__ == "__main__":
    # Load the data
    data_dict = DataFetcher.read_data(files_list=['train', 'dev', 'test'])

    train_data = DataFetcher.parse_conllu(data_dict['train'])
    dev_data = DataFetcher.parse_conllu(data_dict['dev'])
    test_data = DataFetcher.parse_conllu(data_dict['test'])

    # Concatenate all data in one dataframe
    all_data = train_data + dev_data + test_data

    # Create a dataframe with the all words and the true (actual) POS tags.
    all_tuples_df = pd.DataFrame(DataFetcher.pre_processing(all_data), columns=['word', 'y_true'])

    # Calculate counts per tag.
    tf = all_tuples_df.groupby('y_true').count()['word']

    tags_freqs = pd.DataFrame({'tag': tf.index, 'count': tf.values})

    plot_data_stats(tags_freqs)
