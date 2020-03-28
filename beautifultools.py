"""Beautifultools is a module containing all the functions related to the scraping project carried out by Siemens Italy Digital Industries.

Author: Marco Repetto
"""
import statistics, re, string

import nltk, requests 
import numpy as np
import plotly.express as px
import pandas as pd
import nltk.stem as stem
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.linear_model import LinearRegression
from scipy import sparse


def webpage_text_tokenizer(url, body=True, blackL=[""], verbose=False, sep=" ", **kwargs):
    """Request and parse the HTML of a given url. Returns a list of words.
    
    
    Parameters
    ----------
    url : a string pointing to the webpage
    body : a boolean stating whether the whole body should be parsed (default True)
    blackL : a list containing the blacklisted words (default [''])
    verbose : a boolean stating whether the function should print the progresses ()
    sep : the separator for the list of keywords
    """
    
    if verbose:
        print("Checking: " + url)

    # Try the extraction if it fails or timeout is reached return an empty list
    try:
        # Get html of the url
        html = requests.get(url, timeout=10, **kwargs)
        soup = BeautifulSoup(html.text, "html.parser")
        
        # Get webpage Title
        try:
            title = soup.find(name="title").get_text()
        except AttributeError:
            title = ""

        # Get webpage og Title
        try:
            ogTitle = soup.find(name="meta", attrs={"property": "og:title"}).get(
                "content"
            )
        except AttributeError:
            ogTitle = ""

        # Get webpage Keywords
        try:
            keywords = soup.find(name="meta", attrs={"name": "keywords"}).get("content")
        except AttributeError:
            keywords = ""

        # Get webpage Description
        try:
            description = soup.find(name="meta", attrs={"name": "description"}).get(
                "content"
            )
        except AttributeError:
            description = ""

        # Get webpage og Description
        try:
            ogDescription = soup.find(
                name="meta", attrs={"property": "og:description"}
            ).get("content")
        except AttributeError:
            ogDescription = ""

        # If flag body is true then extract all the text in the webpage
        if body:
            # Get webpage Body
            try:
                body = soup.get_text()
            except AttributeError:
                body = ""
        else:
            body = ""
        
        # Wrap and tokenize all the information
        mainCorpora = (
            title
            + " "
            + ogTitle
            + " "
            + keywords
            + " "
            + description
            + " "
            + ogDescription
            + " "
            + body
        )
        token = nltk.word_tokenize(mainCorpora)

        # Make all the tokens lower than trim non words as well as stopwords both in italian an in english
        tokenFiltered = [
            i.lower()
            for i in token
            if not re.search("[" + string.punctuation + "0-9" + "]", i)
            and len(i) >= 3
            and i.lower() not in stopwords.words("english")
            and i.lower() not in stopwords.words("italian")
            and i.lower() not in blackL
        ]

        if verbose:
            print("Number of words found: " + str(len(tokenFiltered)))
        return sep.join(tokenFiltered)

    except:
        if verbose:
            print("Internalerror")
        return ""


def urlize_string(url, warning=True, verbose = False, **kwargs):
    """Normalize the url provided and test whether it can be reached with requests. Returns a string with the corrected url.
    
    
    Parameters
    ----------
    url : a string pointing to the webpage
    warning : a boolean stating whether the function should print the warning
    """

    # Check whether to shutdown warnings
    if not (warning):
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    
    url = url.split(sep=";")
    
    for i in url:
        if verbose:
            print("Processing: ",i)
        try:
            requests.get(i, timeout=10, **kwargs)
            return i
        except requests.exceptions.ConnectionError:
            if verbose:
                print("Connection error with: ",i)
            return np.nan
        except requests.exceptions.ReadTimeout:
            if verbose:
                print("Read timeout error with: ",i)
            return np.nan
        except requests.exceptions.TooManyRedirects:
            if verbose:
                print("Too many redirects error with: ",i)
            return np.nan
        except requests.exceptions.InvalidSchema:
            if verbose:
                print("Invalid schema error with: ",i)
            return np.nan
        except requests.exceptions.MissingSchema:
            newUrl = "HTTP://" + i
            try:
                requests.get(newUrl, timeout=10, **kwargs)
                return newUrl
            except requests.exceptions.ConnectionError:
                if verbose:
                    print("Connection error with: ",newUrl)
                return np.nan
            except requests.exceptions.ReadTimeout:
                if verbose:
                    print("Read timeout error with: ",newUrl)
                return np.nan
            except requests.exceptions.TooManyRedirects:
                if verbose:
                    print("Too many redirects error with: ",newUrl)
                return np.nan
            except requests.exceptions.InvalidSchema:
                if verbose:
                    print("Invalid schema error with: ",newUrl)
                    
                    
def plot_word_frequencies(corpus, **kwargs):
    """Plot the histogram of the words contained into a corpus.
    
    
    Parameters
    ----------
    corpus : a string containing words separated by spaces
    """

    corpus = nltk.tokenize.word_tokenize(corpus)
    corpus = pd.DataFrame({"Word": corpus})

    return px.histogram(corpus, x="Word", **kwargs).update_xaxes(
        categoryorder="total descending"
    )


def drop_duplicates(row, sep=" "):
    """Drop duplicates in a string. It's possible to provide a separator.
    
    
    Parameters
    ----------
    row : a string containing words separated by some separator.
    sep : the separator of words (default ' ')
    """
    
    words = row.split(sep)
    return sep.join(np.unique(words).tolist())


def drop_non_sentiment_words(row, sentiment_words, sep=" "):
    """Drop non sentiment charged words in a string. It's possible to provide a separator.
    
    
    Parameters
    ----------
    row : a string containing words separated by some separator.
    sentiment_words : a list containing the sentiment charged words.
    sep : the separator of words (default ' ')
    """
    
    words = row.split(sep)
    return sep.join([i for i in words if i in sentiment_words])



def stem_words(string, sep=" "):
    """Stem words in a string. The function tries first to stem the word in Italian if nothing happens then switch to English
    
    Parameters
    ----------
    string : a string containing words separated by some separator.
    sep : a separator (default ' ')
    """

    stemmerIta = stem.SnowballStemmer("italian")
    stemmerEng = stem.SnowballStemmer("english")

    string = string.split(sep)
    string = [
        stemmerIta.stem(i) if stemmerIta.stem(i) != i else stemmerEng.stem(i)
        for i in string
    ]

    return sep.join(string)


class MarginalScreening(BaseEstimator):
    """ The estimator implements marginal screening as posed by Kelly et al. 2019 article 'Predicting Returns with Text Data'. The procedure consist of fitting a linear regression for each element of a document word matrix.
    The parameters are controls on the coefficients, upper and lower bound (alpha_plus, alpha_minus) and moreover on the frequency of the words.   
    Parameters
    ----------
    alpha_plus : float, default= 0.5
        A parameter for trimming the coefficients downwards.
    alpha_minus : float, default= 0.5
        A parameter for trimming the coefficients upwards.
    kappa : float, default= 1
        Lower limit on words frequency.
    """
    def __init__(self, alpha_plus = 0.5, alpha_minus = 0.5, kappa = 1):
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.kappa = kappa

    def fit(self, X, y):
        """Implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True, y_numeric=True)
        
        wordfreq = sum(X)
        wordfreq = wordfreq.toarray()[0,:]
        
        self.coef_ = np.array([])
        
        # Loop for every column in the matrix
        for i in X.T:
            coefficient = LinearRegression(fit_intercept=False).fit(i.T,y).coef_
            self.coef_ = np.concatenate((self.coef_, coefficient))
            
        # Filter the coefficients based on the parameters
        self.coef_[(self.coef_ < self.alpha_plus) * (self.coef_ > self.alpha_minus)] = np.nan
        self.coef_[wordfreq < self.kappa] = np.nan
        
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        
        yHat = np.array([])

        for i in range(0, X.shape[0]):
            # Mean voting
            yHatElement = np.nan
            yHatElement = np.nanmean(self.coef_[(X[i, :].toarray()[0] == 1) * (np.isfinite(self.coef_))])
            yHat = np.concatenate((yHat,np.array([yHatElement])))
        
        return yHat


class TopicModelling(BaseEstimator):
    """ The estimator implements marginal screening as posed by Kelly et al. 2019 article 'Predicting Returns with Text Data'. The procedure consist of fitting a linear regression for each element of a document word matrix.
    The parameters are controls on the coefficients, upper and lower bound (alpha_plus, alpha_minus) and moreover on the frequency of the words.   
    Parameters
    ----------
    """
    def __init__(self, rank=True):
        self.rank = rank

    def fit(self, X, y):
        """Implementation of a fitting function.
        Parameters
        ----------
        X : document word matrix of sentiment charged words 
        y : orders 
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True, y_numeric=True)
        
        s = np.sum(X, axis = 1)
        
        tildeD = X/s[:,None]
        
        self.coef_ = tildeD
        
        return self

