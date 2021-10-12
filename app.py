#!/usr/bin/env python3

import streamlit as st

import basic
import clustering
import get_data
import prediction


def main():
    df = get_data.df
    st.header('Subscription Prediction')
    st.subheader('Data Exploration')
    st.markdown('''First, let's take a look at the data. We can see the first rows of the data:''')
    st.write(df.head(100))
    st.markdown('''We can see that we have data about the times of actions such as installation, subscription and 
    feature usage; behavioral usage and personal data such as features used, acceptance, time of usage, country etc and 
    whether the user subscribed or not. 
    \nIn the next table we can see the number of unique values in every column:''')

    st.write(df.nunique())
    st.markdown('''We can already see that the data is clustered at the user and session levels. i.e, 
    every row represents a feature used by user at a specific instance. The dataset contains data on 6177 users. 
    On average a user uses the app for 1.5 sessions and each sessions composes of 6 features use. 
    In order to dive deeper into the data let's look
    at the values of the categorical features:''')

    cols = st.selectbox('columns', ['country', 'feature_name', 'device'])

    unique_val = basic.get_unique_values(cols)
    st.write(unique_val)

    st.markdown('''In order to get a better understanding of the different features let's take a look at the histograms
    of the different features:''')
    cols = st.selectbox('columns', ['device_timestamp', 'feature_name', 'device', 'accepted', 'usage_duration',
                                    'install_date', 'country', 'device', 'subscription_date', 'subscriber'])

    histograms = basic.plot_hist(df, cols)
    st.plotly_chart(histograms)

    st.markdown('''We can notice some interesting aspects of the data: 
    Some app features are much more popular than others;
    specifically, relight, backdrop and touch up are the most popular features in the app. Most of the use is being done
    in mobiles rather than tablets. Only minority of the changes are accepted by the users, about 66% denied. The use in 
    every feature in the app is limited to a few seconds in most cases. The app is most popular in the US and in Europe.
    \nThe most important fact we can extract from these plots is the imbalance of the data - 
    only a small fraction of the 
    users are subscribers (from this plot we can't tell exactly how much since we need to group the data in a 
    user level. 
    We will do it soon).
    \nTo better understand our data we would like to learn more about the relation between the different features.
    In the next tab we will investigate this angle deeper. We will create a pivot table with different groupings. 
    By choosing features in the first two select boxes we group our data. We can observe the 
    data from the user perspective
    angle as well as from the session angle. The operation on the data can be sum or mean for numeric features and count
    for categorical features.''')

    indices = st.multiselect('group by indices', ['id_for_vendor', 'device_timestamp', 'feature_name',
                                                  'app_session_id', 'accepted', 'usage_duration', 'install_date',
                                                  'country', 'device', 'subscription_date', 'subscriber'],
                             default=['feature_name'])

    cols = st.multiselect('group by cols', ['id_for_vendor', 'device_timestamp', 'feature_name',
                                            'app_session_id', 'accepted', 'usage_duration', 'install_date',
                                            'country', 'device', 'subscription_date', 'subscriber'],
                          default=['accepted'])

    clusters = st.selectbox('clusters', [None, 'id_for_vendor', 'app_session_id'], index=1)

    agg_func = st.selectbox('function', ['mean', 'sum', 'count'], index=2)
    fig, table = basic.plot_pivot(indices, cols, agg_func=agg_func, values=None, cluster_unit=clusters)
    st.write(table.head(100))
    st.plotly_chart(fig)

    st.markdown('''We can notice a few insightful things. First, by looking at the connection between the 
    "feature_names" and "accepted" we can see that though most changes being denied some are more popular than others. 
    Features such as "face" and "retouch" have high acceptance rate while "paint", "reshape" and 
    "vignette" have almost no acceptance. Maybe it 
    can teach us that many users use the app as a quick editing tool before posting rather than a profound tool
    for editing photos (such as Adobe's Lightroom/Photoshop which demand higher expertise).
    \nBy observing the group of "subscriber" ("subscriber"-None-None-mean") 
    we can see some differences between subscribers
    and non; Subscribers accept more changes and spend less time on each feature on average. 
    Due to the small amount of subscribers the data may not reflect the true estimations. 
    \nSo, how much users subscribe? By looking at the subscribers ("subscribers"-None-None-Count") we can see that
    there are 6129 "id_for_vendor" (i.e users) that didn't subscribed and only 48 that did. 
    We can also see that subscribers
    only come from 15 countries and that .
    \nThere are many more aspects that we can cover. e.g, how much time it takes to a subscriber to subscribe etc.''')

    st.subheader('Feature Engineering')
    st.markdown('''Thus far we analyzed some aspects of the data. In order to train a model we need numeric, 
    informative features. The problem in the current format of the data is that it is mostly uninformative. 
    Many features are sparse categories and some are mere date features. We will now go over problems with the current 
    structure of the data and offer some changes.
    \n+ The **"id_for_vendor"** feature is our cluster id - 
    our prediction should indicate whether every cluster (i.e, user) will 
    subscribe. The problem is that we have multiple sessions per cluster 
    so we would inflate our estimators were we using 
    every row independently. There are two options regarding this problem; 
    we can "flat" the data by engineering every user
    to one observation. For instance, instead of one observation per feature 
    use we can count the total use of every feature
    by a single user and use it as a feature for our model. 
    Our second alternative is to use mixed models. In these models
    we bring into consideration the clustered structure of our data 
    (we will get back to this topic in the models section).
    \n + The **"Country"** feature indicates the user's country. As we saw earlier there are only 15 (out of 126) 
    countries from which users subscribed. This feature is very sparse 
    and probably doesn't carry many information within.
    In order to make it more informative we can replace it with other 
    informative features of the country, such as the GDP 
    per capita, social media usage etc. We can also make it more dense 
    by using some aggregation (continent, region etc.).
    \n + The **"Device"** feature can also be to sparse and harm our models' performance. We can binarize it to mobiles
    and tablets. This feature can possibly reveal different uses in the app.
    \n + There are multiple columns indicating different time aspects 
    - **"device_timestamp"** indicating the time a feature
    was used, **"usage_duration"** indicates the length of the use in a feature, 
    **"install_date"** indicates the time of
    installation and **"subscription_date"** is obviously indicates the time of subscription. We will use the 
    **"usage_duration"** feature to indicate the importance of the feature. 
    Regarding the installation and subscription - 
    the logical thing to do would be to use it in a survival analysis; some of the non-subscribers are actually a 
    not-yet-subscribed. The model should take this into account and for that a "time to subscription" feature is needed. 
    \n + The **"feature_name"** feature indicates what app features were used. 
    As we saw earlier there are 17 features with 
    varying popularity measures. We can simply one-hot-encode it, 
    but we want to take into account the intensity of the use
    so we will use the **"accepted"** feature and the **"usage_duration"** combined with it.
    \nWe mentioned two options regarding the clusters - 
    either use mixed models or average somehow the different sessions 
    into one observation. The data for mixed model will look as follows:  
    ''')

    st.write(basic.basic_editing().head())

    st.markdown('''We can see that we have multiple observations per user yet. This will be the grouping unit data in a
    mixed model. 
    \n Next we can see the averaged data. There are several ways to calculate the intensity of the usage in every app 
    feature. We can simply count the number of uses a user use a feature, 
    we can multiply it with the duration she used it 
    or count only the accepted changes (we can, but won't, combine the different approaches). 
    In the following tab we can compare the different approaches:
    ''')

    app_features_type = st.selectbox('type of calculation', ['acceptance', 'duration', 'additive'], index=2)
    st.write(basic.piping(app_features_type).head())

    st.subheader('Models')
    st.markdown('''In this part we will train different models and evaluate them.''')
    st.markdown('#### Imbalance')
    st.markdown('''We have seen earlier that only 48 users 
    in our data are subscribers. It makes it "beneficial" for models to
    predict majority cases. To demonstrate the point let's train a simple logistic regression model and examine its 
    results. In the following tables we can see the confusion matrix and a classification report:''')

    st.write(prediction.confusion_basic_log)
    st.text(prediction.report_basic_log)

    st.markdown('''As we can see the results are miserable - no case was predicted as positive at all, 
    and all of the parameters of the positive 
    class - zero precision, recall and f1-score - are zero. Yet the average precision is very high. 
    In order to deal with this problem
    we will use oversampling, undersampling, different scoring and specialized models. 
    We will use logistic regression and random forest 
    as our models, though we can possibly try other models. The advantage in logistic regression is its simplicity, 
    but it's come at the price of strong assumption of linearity. Random forest can capture some nonlinearities in the
    data but it's less simple. We will train the models with 5-fold cross validation with f1 and balanced accuracy 
    as our metrics. The use of these metrics is due to the imbalance. 
    Focusing on accuracy will cause our model to neglect
    the minority class. 
    \nIn the next table we can see the mean results of the different models:
    ''')
    x, y = basic.split(basic.piping('additive'), 'cv')
    report = prediction.running_models(x, y)

    st.write(report)
    st.markdown('''As we can see the performance is bad, most of the classifiers are as good as random. 
    Some of the used techniques display minor improvement. We can take a closer look at the best model - 
    logistic regression with smote. In the next two tables we can see the confusion matrix and a classification report.
    We can see that the recall is much higher since the model gets some sense of the positive class. 
    Yet the precision is 
    very low.''')

    st.markdown('#### Additional Steps')
    st.markdown('''We could do some additional steps to improve and understand our analysis better. 
    First, we can improve
    our models by optimizing the parameters of a chosen model. We can use grid search (or other kind of search) 
    to find better parameters. A feature selection process is also advisable. Model like "RFE" 
    (Recursive Feature Elimination) can help us with that task. To better understand our model we could add ROC curve 
    and measure the AUC. Finally, we could do error analysis to find where our models work best and what are their blind 
    spots. ''')

    st.header('Clustering')
    st.markdown('''In this part we will try to find clusters of users from our data. As usual we will start with some 
    basic description of the data. In the next table we can see the head of the data:''')
    from get_data import df_cluster
    st.write(df_cluster.head())

    cols_cluster = st.selectbox('column name', [i for i in df_cluster.columns], index=3)

    histograms = basic.plot_hist(df_cluster, cols_cluster)
    st.plotly_chart(histograms)
    st.markdown('''We can see that most features declines over time - 
    most of the users use for a short amount of time. We 
    can also see that only 3 networks are significant, 
    and there are a small amount of significant devices. It can help the
    clustering but we will drop these variables now. 
    \nWe will take two paths in order to find clusters in our data. We will try the "classic" algorithm of k-means, 
    and we will also decompose the data using SVD.''')

    st.markdown('#### KMeans')

    st.markdown(''''\nFirst we will train a k-means model with different number of clusters. 
    Since we don't have any ground truth we will use the "elbow" method with inertia and silhouette. 
    The following plot shows the inertia from 1 to ten clusters.''')

    clus, elbow, fitted, pred = clustering.running_kmeans()
    st.write(elbow)
    st.caption('''There is a problem with the axes labels. It should be reversed.''')
    st.markdown('''It is not very clear in this case but we will take 6 as the number of clusters. 
    We can also check what
    is the silhouette score of different cluster numbers. The inertia shows us how "tight" a class is, 
    while a silhouette
    shows us whether or not the separation between classes are good. 
    If similar users are close together and dissimilar
    users are further away. We would also want to observe the clustering visually. Since it's high dimensional data we 
    can't do it directly so we would simplify it first by using PCA and then we would plot it:''')

    st.plotly_chart(clustering.vis_kmeans(clus, pred))

    st.markdown('''From this plot it would be hard to see whether our clustering is good enough. There is no clear 
    pattern of clustering in the data. The next thing we would like to do is taking it back to the data. We can do 
    different things to understand what our clustering means, but we would do the following thing: we would look at the
    centroids of our 6 clusters:''')

    regex = st.text_input('column name (regex)', 'first_day.*(sessions|projects)|(since)')
    st.plotly_chart(clustering.plot_result_kmeans(clus, fitted, regex))

    st.markdown('''Every line is a centroid of a cluster. 
    In this particular plot we are examining the connection between
    the usage intensity in the first day and the time since the last use. We can see some connection - those who weren't
    excited in the first use leave the app and doesn't come back. The opposite is also true - the excited ones stick 
    with the app. We can check many more aspects in the same manner - are early adopters use more pro features? 
    can we identify whether are there specific features used by professional advertisers/influencers? 
    \nWe can use a new feature to answer the last question - the sharing app. Given edited photo we can see what app it 
    was shared with. We can expect to see different patterns among the users who shared their photos via whatsapp and
    instagram. This feature can help us with the identification of amateurs and professionals.''')

    st.markdown('#### SVD')
    st.markdown('''I would suggest another method for clustering, though I will not implement it here. 
    We can use "Singular Value Decomposition" (SVD) for this task. The data we have is showing some preferences for each 
    user. SVD decomposes the data to 3 conceptually different parts. 
    The first is the connection between the columns in the 
    data to abstract concepts. The third part is the connection between the users and those abstract concepts. 
    The second part measures the strength of a concept. For example, 
    if a "selfie" is a photography concept SVD can allow us
    to find the features in the app that go hand in hand with selfie. 
    It would also allow us to see which of our users is 
    prone to selfies.''')


if __name__ == '__main__':
    main()
