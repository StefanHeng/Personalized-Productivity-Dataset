import os
import pandas as pd


if __name__ == '__main__':
    from stefutil import *

    def check_myca_eg():
        """
        Check the myca CSV dataset for example usage

        Would assume those are by a single user
        """
        path = os.path.join('..', 'dataset', 'categories_user_4 - categories_user_4.csv')
        df = pd.read_csv(path)
        mic(df)

        main_groups = df[df.type == 'workset'].title
        mic(main_groups)
    # check_myca_eg()

    def check_stackoverflow():
        import tensorflow_federated as tff  # installing on python3.10 fails
        dset = tff.simulation.datasets.stackoverflow.load_data()
        tr, vl, ts = dset
        mic(tr, vl, ts)
        mic(len(tr.client_ids))
        cid = tr.client_ids[0]
        tr = tr.create_tf_dataset_for_client(cid)
        mic(tr)
    check_stackoverflow()

    def check_reddit():
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'try-reddit_key.json'

        # q = """
        # SELECT r.body, r.score_hidden, r.name, r.author, r.subreddit, r.score
        # FROM `fh-bigquery.reddit_comments.2019_08` r
        # WHERE r.subreddit = "dataisbeautiful"
        # and r.body != '[removed]'
        # and r.body != '[deleted]'
        # LIMIT 20
        # """
        q = """
        SELECT * FROM fh-bigquery.reddit_comments.2019_08
        LIMIT 20
        """

        # Submit and get the results as a pandas dataframe
        mic('loading')
        df = pd.read_gbq(q, project_id='try-reddit')
        mic(df)
    # check_reddit()
