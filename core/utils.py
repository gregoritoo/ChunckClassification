import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb
import umap
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import catboost as cb


def preprocess_text(text):
    """
    Preprocesses a given text by tokenizing, filtering stopwords, and lemmatizing.

    This function performs the following steps on the input text:
    1. Tokenizes the text using a regular expression tokenizer.
    2. Converts tokens to lowercase.
    3. Removes stopwords from the tokens.
    4. Lemmatizes the tokens using the WordNet lemmatizer.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        list: A list of preprocessed and lemmatized tokens.

    Example:
        input_text = "The quick brown foxes are jumping over the lazy dog."
        preprocessed_tokens = preprocess_text(input_text)
    """
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text.lower())
    filtered_tokens = [
        token for token in tokens if token not in stopwords.words("english")
    ]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return lemmatized_tokens


def do_basic_analysis(df):
    """
    Perform basic analysis on a DataFrame.

    This function provides basic statistical analysis and information about the given DataFrame,
    including summary statistics and the number of null elements in the dataset.

    Args:
        df (pandas.DataFrame): The input DataFrame to be analyzed.

    Returns:
        None

    Prints:
        - Summary statistics of the DataFrame using df.describe().
        - The total number of null elements in the dataset.
        - For each column with null elements, the number of null elements in that column.

    Example:
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, None, 6], 'C': [7, 8, None]})
        do_basic_analysis(df)
    """
    print(df.describe())
    print(f"Number of null elemnt in dataset : {df.isnull().values.sum()}")
    for column in df.keys():
        if df[column].isnull().values.any():
            print(f"{df[column].isnull().values.sum()} element null in column {column}")


def do_disribution_analysis(df):
    """
    Perform distribution analysis and visualize histograms for numerical columns in a DataFrame.

    This function calculates the histograms for numerical columns in the given DataFrame and generates
    a subplot of histograms to visualize the distribution of values.

    Args:
        df (pandas.DataFrame): The input DataFrame containing both numerical and non-numerical columns.

    Returns:
        None

    Example:
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': ['X', 'Y', 'Z']})
        do_distribution_analysis(df)
    """
    nb_columns_splitter = 2
    columns_non_numerical = find_non_numerical_columns(df)
    columns = [column for column in df.keys() if column not in columns_non_numerical]
    fig, ax = plt.subplots(
        nb_columns_splitter, int(len(columns) / nb_columns_splitter), figsize=(20, 7)
    )
    for pos, column in enumerate(columns):
        labels = df[column].values
        counts, edges, bars = ax[
            pos % nb_columns_splitter, pos // nb_columns_splitter
        ].hist(labels, bins=10)
        ax[pos % nb_columns_splitter, pos // nb_columns_splitter].bar_label(bars)
        ax[pos % nb_columns_splitter, pos // nb_columns_splitter].set_title(
            f"Distribution {column}"
        )
    plt.show()


def do_correlation_analysis(df):
    """
    Perform correlation analysis and visualize correlation matrix for numerical columns in a DataFrame.

    This function calculates the correlation matrix for numerical columns in the given DataFrame and
    generates a heatmap visualization of the correlations.

    Args:
        df (pandas.DataFrame): The input DataFrame containing both numerical and non-numerical columns.

    Returns:
        None

    Example:
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': ['X', 'Y', 'Z']})
        do_correlation_analysis(df)
    """
    non_numerical_columns = find_non_numerical_columns(df)
    for column in non_numerical_columns:
        restricted_df = categorize_columns(restricted_df, column, f"{column}_cat")
    numerical_df = restricted_df.drop(columns=non_numerical_columns)
    corr_matrix = numerical_df.corr()
    plt.figure(figsize=(15, 8))
    sn.heatmap(corr_matrix, annot=True)
    plt.show()


def categorize_columns(df, origin_column_name, target_column_name):
    """
    Caterogize wanted columns
    Args :
        df (pandas dataframe),  original dataframe
        origin_column_name (list or str), name or list of names of columns to categorize
        origin_column_name (list or str), name or list of names of columns where
                                          to add categorized columns
    Outs :
        df (pandas dataframe), dataframe with added columns
    """
    if isinstance(origin_column_name, str):
        assert isinstance(target_column_name, str)
        origin_column_name = [origin_column_name]
        target_column_name = [target_column_name]

    for colum_name, target_name in zip(origin_column_name, target_column_name):
        df[colum_name] = pd.Categorical(df[colum_name])
        df[target_name] = df[colum_name].cat.codes
    return df


def convert_df_to_numeric_df(df, columns_to_drop):
    """
    Convert a DataFrame to a numeric-only DataFrame by dropping specified columns and categorizing non-numerical columns.

    This function creates a new DataFrame by dropping specified columns and categorizing non-numerical columns
    in the input DataFrame. The resulting DataFrame contains only numerical features.

    Args:
        df (pandas.DataFrame): The input DataFrame containing a mix of numerical and non-numerical columns.
        columns_to_drop (list): List of column names to be dropped from the DataFrame.

    Returns:
        pandas.DataFrame: A new DataFrame containing only numerical features after dropping specified columns
        and categorizing non-numerical columns.

    Example:
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['X', 'Y', 'Z'], 'C': [4.5, 6.7, 8.9]})
        columns_to_drop = ['B']
        numeric_df = convert_df_to_numeric_df(df, columns_to_drop)
    """
    try:
        df = df.drop(columns=columns_to_drop)
    except Exception as e:
        pass
    non_numerical_columns = find_non_numerical_columns(df)
    for column in non_numerical_columns:
        df = categorize_columns(df, column, f"{column}_cat")
    df = df.drop(columns=non_numerical_columns)
    return df


def find_non_numerical_columns(df):
    """
    Finds non-numerical columns in a DataFrame.

    This function scans each column of the input DataFrame and identifies columns that contain non-numerical
    (string) values. It checks if the first element in the column is a string, and then it further checks if any
    non-numeric string values exist in the column.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list of column names containing non-numerical values.

    Example:
        input_df = pd.DataFrame({'col1': ['apple', 'banana', 'cherry'],
                                 'col2': [123, '456', '789']})
        non_numerical_columns = find_non_numerical_columns(input_df)
    """
    non_numerical_columns = []
    for column in df.keys():
        if isinstance(df[column][0], str):
            if sum(
                [row.isnumeric() for row in df[column].values if isinstance(row, str)]
            ) != len(df[column]):
                non_numerical_columns.append(column)
    return non_numerical_columns


def scale_columns(df, columns=None, return_scaler=False):
    """
    Scales specified columns of a DataFrame using StandardScaler.

    This function scales the specified columns of the input DataFrame using the StandardScaler from scikit-learn.
    The specified columns are transformed to have zero mean and unit variance.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list, optional): List of column names to scale. If None, all columns will be scaled.

    Returns:
        pd.DataFrame: A new DataFrame with scaled columns.

    Example:
        input_df = pd.DataFrame({'col1': [1, 2, 3],
                                 'col2': [4, 5, 6]})
        scaled_df = scale_columns(input_df, columns=['col1', 'col2'])
    """
    keys_to_keep = [
        column_name for column_name in df.keys() if column_name not in columns
    ]
    std_scaler = StandardScaler()
    sub_df = df[columns].copy()
    df_scaled = std_scaler.fit_transform(sub_df.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=columns)
    for key in keys_to_keep:
        df_scaled[key] = df[key]
    if not return_scaler:
        return df_scaled
    else:
        return df_scaled, std_scaler


def evaluate_model(X_test, y_test, model, verbose=0):
    """
    Evaluate the performance of a binary classification model using various metrics.

    This function takes the input test data `X_test` and corresponding ground truth labels `y_test`,
    along with a binary classification `model`, and evaluates its performance using various metrics
    including balanced accuracy, AUC-ROC curve, and a confusion matrix.

    Args:
        X_test (array-like): Input test data.
        y_test (array-like): Ground truth labels for the test data.
        model: Binary classification model with a `predict` method.
        verbose (bool, optional): Whether to print evaluation metrics and display confusion matrix. Default is True.

    Returns:
        float: Balanced accuracy score.

    Example:
        X_test_data = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        y_test_labels = [0, 1, 0]
        trained_model = ...  # Train a binary classification model
        evaluate_model(X_test_data, y_test_labels, trained_model)
    """
    y_predicted = model.predict(X_test)
    y_predicted = np.where(y_predicted > 0.5, 1, 0)
    fpr, tpr, thresholds = roc_curve(y_test, y_predicted, pos_label=1)
    if verbose == 1:
        print(f" balanced accuracy is {balanced_accuracy_score(y_test,y_predicted)}")
        print(f" AUC is {auc(fpr, tpr)}")
        c_matrix = confusion_matrix(y_predicted, y_test)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=c_matrix, display_labels=np.unique(y_test)
        )
        disp.plot()
    return balanced_accuracy_score(y_test, y_predicted)


def evaluate_cross_validation(scaled_df, model, target, cv=5, test_size=0.2):
    """
    Evaluate a binary classification model using cross-validation.

    This function performs cross-validation for evaluating a binary classification model's performance.
    It takes the input data `scaled_df`, the binary classification `model`, the target labels `target`,
    the number of cross-validation folds `cv`, and the test data size `test_size` for each fold.

    Args:
        scaled_df (pandas DataFrame): Scaled input data.
        model: Binary classification model to be evaluated.
        target (array-like): Target labels corresponding to the input data.
        cv (int, optional): Number of cross-validation folds. Default is 5.
        test_size (float, optional): Proportion of the data to include in the test split. Default is 0.2.

    Returns:
        list: List of balanced accuracy scores for each fold.

    Example:
        scaled_data = ...  # Scaled input data as a pandas DataFrame
        target_labels = [0, 1, 0, ...]  # Target labels for the input data
        classification_model = ...  # Instantiate a binary classification model
        scores = evaluate_cross_validation(scaled_data, classification_model, target_labels)
    """
    ranndom_states = [random.randint(0, 10) for _ in range(cv)]
    balanced_accuracys = []
    for random_state in ranndom_states:
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_df,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )
        model.fit(X_train, y_train)
        balanced_accuracys.append(evaluate_model(X_test, y_test, model, verbose=False))
    return balanced_accuracys


def objective_xgb(xgb_model, params, X_train, y_train, X_test, y_test):
    """
    Optimize XGBoost hyperparameters using cross-validation.

    This function optimizes the hyperparameters of an XGBoost classifier using cross-validation.
    It takes an initial XGBoost classifier model `xgb_model`, a dictionary of hyperparameters `params`,
    training data `X_train` and `y_train`, and testing data `X_test` and `y_test`.

    Args:
        xgb_model: Initial XGBoost classifier model.
        params (dict): Dictionary of hyperparameters for XGBoost.
        X_train (array-like): Training data features.
        y_train (array-like): Training data target labels.
        X_test (array-like): Testing data features.
        y_test (array-like): Testing data target labels.

    Returns:
        dict: Dictionary containing the optimization result with loss value and status.
    """
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    score = evaluate_model(X_test, y_test, xgb_model)
    return {"loss": -score, "status": 200}


def clustering(vectorize_sentence, words_embeddings):
    """
    Perform dimensionality reduction and clustering using UMAP.

    This function performs dimensionality reduction and clustering on input sentence vectors
    and word embeddings using the UMAP algorithm. It takes a matrix of sentence vectors
    `vectorize_sentence` and a matrix of word embeddings `words_embeddings`.

    Args:
        vectorize_sentence (array-like): Matrix of sentence vectors.
        words_embeddings (array-like): Matrix of word embeddings.

    Returns:
        tuple: A tuple containing two arrays. The first array is the reduced representation
        of sentence vectors using UMAP. The second array is the reduced representation of
        word embeddings using UMAP.
    """
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(vectorize_sentence)
    embedding_words = reducer.transform(words_embeddings)
    return embedding, embedding_words


def get_topic_description(centroids, embedding_words):
    """
    Generate topic descriptions based on centroids and word embeddings.

    This function generates topic descriptions using the provided centroids and word embeddings.
    It calculates the similarity between centroids and word embeddings and returns a dictionary
    containing centroid indices and indices of top-k similar word vectors for each topic.

    Args:
        centroids (array-like): Centroids representing topics.
        embedding_words (array-like): Matrix of word embeddings.

    Returns:
        dict: A dictionary where keys are topic names and values are dictionaries containing
        centroid index and top-k similar word vector indices for each topic.

    """
    dic_topics = {}
    for topic in range(centroids.shape[0]):
        dic_topic = {}
        similarity = np.matmul(
            embedding_words, centroids[topic, :].reshape(1, -1).T
        ).mean(axis=1)
        ordered_similarity = np.argsort(similarity)
        dic_topic.update({"centroid": ordered_similarity[0]})
        dic_topic.update({"top k vectors": ordered_similarity[1:10]})
        dic_topics.update({"topic_" + str(topic): dic_topic})
    return dic_topics


def print_topic(topic, dic_topics, dictionnary):
    """
    Print the description of a topic.

    This function prints the description of a topic using the provided topic index,
    dictionary of topic descriptions, and a word-to-index dictionary.

    Args:
        topic (int): Index of the topic to print.
        dic_topics (dict): Dictionary containing topic descriptions.
        dictionary (dict): Word-to-index dictionary.

    Returns:
        None
    """
    print(dictionnary[dic_topics["topic_" + str(topic)]["centroid"]])
    print(
        [
            dictionnary[word]
            for word in dic_topics["topic_" + str(topic)]["top k vectors"]
        ]
    )


def optimize_xgb(X_train, y_train, X_test, y_test):
    params = {
        "max_depth": hp.choice("max_depth", np.arange(2, 12, dtype=int)),
        "learning_rate": hp.loguniform("learning_rate", -10, 10),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "scale_pos_weight": hp.choice(
            "scale_pos_weight", [1, 10, 25, 50, 75, 99, 100, 1000]
        ),
    }

    def objective_xgb_note(params):
        xgb_model = xgb.XGBClassifier(
            **params, tree_method="gpu_hist", enable_categorical=True
        )
        xgb_model.fit(X_train, y_train)
        score = evaluate_model(X_test, y_test, xgb_model, verbose=False)
        return {"loss": -score, "status": STATUS_OK}

    best_params = fmin(objective_xgb_note, params, algo=tpe.suggest, max_evals=100)
    print("Best set of hyperparameters: ", best_params)
    xgb_model = xgb.XGBClassifier(**best_params)
    xgb_model.fit(X_train, y_train)
    score = evaluate_model(X_test, y_test, xgb_model, verbose=True)
    return xgb_model


def optimize_catboost(X_train, y_train, X_test, y_test):
    params = {
        "depth": hp.choice("max_depth", np.arange(2, 12, dtype=int)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", 1, 10),
        "iterations": hp.choice("iterations", np.arange(50, 150, dtype=int)),
        "learning_rate": hp.uniform("learning_rate", 0.001, 0.1),
        "verbose": 0,
    }

    def objective_xgb_note(params):
        cat_model = cb.CatBoostClassifier(
            **params, loss_function="Logloss", eval_metric="AUC"
        )
        # cat_model = xgb.XGBClassifier(**params,tree_method="gpu_hist", enable_categorical=True)
        cat_model.fit(X_train, y_train)
        score = evaluate_model(X_test, y_test, cat_model, verbose=False)
        return {"loss": -score, "status": STATUS_OK}

    best_params = fmin(objective_xgb_note, params, algo=tpe.suggest, max_evals=10)
    print("Best set of hyperparameters: ", best_params)
    cat_model = cb.CatBoostClassifier(**best_params)
    cat_model.fit(X_train, y_train, verbose=0)
    score = evaluate_model(X_test, y_test, cat_model, verbose=True)
    return cat_model
