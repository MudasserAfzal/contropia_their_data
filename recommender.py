import pandas as pd
import numpy as np
import pandas as pd
import json
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def pickle_files_saving():

    df = pd.read_csv("new_data_final_dataframe.csv")
    for i in range(len(df)):
        input_string = df["line_items"][i].replace("\"", "'")
        json_string = input_string.replace("'", "\"").replace('"r ', ', ')
        list_of_dicts = json.loads(json_string)
        df["line_items"][i] = list_of_dicts[0]

    expanded_df = pd.json_normalize(df['line_items'])

    # Concatenate the original DataFrame and the expanded DataFrame
    result_df = pd.concat([df.drop('line_items', axis=1), expanded_df], axis=1)

    # Reset the index if needed
    result_df.reset_index(drop=True, inplace=True)

    # df["line_items"] = df["line_items"].apply(literal_eval)
    # df = df.explode("line_items")
    # df = pd.concat([df, df.pop("line_items").apply(pd.Series)], axis=1)
    # df.drop(['shipping_address'], axis=1, inplace=True)
    # day_list = df["product_id"].unique().tolist()
    # df["product_id"] = np.random.choice(day_list, size=len(df))
    result_df['customer_no'] = np.random.randint(1,10, size=len(result_df))
    # df.drop_duplicates(subset=['product_id'], keep='last', inplace=True)
    result_df = result_df.astype({'customer_no':'int', "product_id":'str', 'gross_price':'float'})

    data = result_df.copy()
    # convert customer_no to int
    data['customer_no'] = data['customer_no'].astype(int)

    data.to_csv('processed.csv', index=False)
    
    # define the reader for the surprise package
    reader = Reader(rating_scale=(data['gross_price'].min(), data['gross_price'].max()))

    # create the surprise dataset
    surprise_data = Dataset.load_from_df(data[['customer_no', 'product_id', 'gross_price']], reader)

    # fit the SVD model on the surprise dataset
    svd = SVD()
    trainset = surprise_data.build_full_trainset()
    svd.fit(trainset)

    # # # create the content-based recommender using the product name
    tfidf = TfidfVectorizer(stop_words='english')
    product_matrix = tfidf.fit_transform(data['product_id'].unique())
    cosine_sim = cosine_similarity(product_matrix, product_matrix)


    import pickle

    # Save SVD model
    with open('svd_model.pkl', 'wb') as f:
        pickle.dump(svd, f)

    # Save content-based recommender models
    with open('tfidf_model.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    with open('cosine_sim_model.pkl', 'wb') as f:
        pickle.dump(cosine_sim, f)

    return svd, tfidf, cosine_sim