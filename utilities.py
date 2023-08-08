import numpy as np

# define a function to get the top N similar products
def get_similar_products(data, cosine_sim, product_id, N=10):
    unique_products = data['product_id'].unique()
    idx = np.where(unique_products == product_id)[0][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    product_indices = [i[0] for i in sim_scores]
    return data.iloc[product_indices]['product_id'].tolist()

# define a function to get the top N recommended products for a given customer
def get_top_recommendations(data, svd, cosine_sim, customer_id, N=10):
    data = data.astype({'customer_no':'int', "product_id":'str', 'gross_price':'float'})
    # get the list of products the customer has already purchased
    customer_products = data[data['customer_no'] == int(customer_id)]['product_id'].tolist()
#     print(customer_id, customer_products)
    
    # get the predicted ratings for all products
    all_products = data['product_id'].unique()

    print(customer_products, all_products)
    testset = [(customer_id, product_id, 4.) for product_id in all_products if product_id not in customer_products]
    predictions = svd.test(testset)
    predicted_ratings = [(p.iid, p.est) for p in predictions]
    
    # sort the predicted ratings by highest to lowest and get the top N recommended products
    top_recs = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:N]
    top_products = [r[0] for r in top_recs]
    
    # get the top N similar products for each top recommended product and remove duplicates
    similar_products = []
    for product_id in top_products:
        similar_products += get_similar_products(data, cosine_sim, product_id, N)
    similar_products = list(set(similar_products))
    
    # remove products the customer has already purchased from the top recommended and similar products
    top_recs = [p for p in top_products if p not in customer_products][:N]
    similar_products = [p for p in top_recs
                        if p not in customer_products][:N]
    
    # return the top N recommended and similar products
    return {'recommended': top_recs, 'similar': similar_products}