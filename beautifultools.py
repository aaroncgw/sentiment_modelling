"""Beautifultools is a module containing all the functions related to the scraping project carried out by Siemens Italy Digital Industries.

Author: Marco Repetto
"""
import statistics, re, string, sys, os

import nltk, requests 
import numpy as np
import plotly.express as px
import pandas as pd
import nltk.stem as stem
from sklearn.preprocessing import normalize
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from scipy.optimize import fmin
from scipy import sparse

def blockPrint():
    """ Disable printing
    """
    
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    """ Enable printing
    """
    sys.stdout = sys.__stdout__

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


class SSESTM(BaseEstimator):
    """ The estimator implements the Supervised Sentiment Extraction via Screening and Topic Modelling (aka SSESTM procedure) as posed by Kelly et al. 2019 article 'Predicting Returns with Text Data'. The procedure consist of:
    1. Marginal screening: fitting a linear regression for each element of a document word matrix;
    2. Topic modelling: fitting a linear regression on the screened words frequencies passing ranked labels;
    3. Prediction: by maximizing the log likelihood of a multinomial distribution with penalty.
 
    The parameters are controls on the coefficients, upper and lower bound (alpha_plus, alpha_minus), on the frequency of the words and on the penalty applied in prediction.   

    Parameters
    ----------
    alpha_plus : float, default= 0.5
        A parameter for trimming the coefficients downwards.
    alpha_minus : float, default= 0.5
        A parameter for trimming the coefficients upwards.
    kappa : float, default= 1
        Lower limit on words frequency.
    l : float, default= 0
        Maximum likelihood penalty term.

    """
    def __init__(self, alpha_plus = 0.5, alpha_minus = 0.5, kappa = 1, l = 0.0):
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.kappa = kappa
        self.l = l
        
    def fit(self, K, y):
        """Implements the first two steps, namely:
        1. Marginal Screening;
        2. Topic Modelling
.
        Parameters
        ----------
        K : pandas series 
            A series containing the keywords.

        y : pandas series
            A series containing labels 

        Returns
        -------
        self : object
            Returns self.
        """

        # MARGINAL SCREENING #
        # Init the document term matrix
        cvStep1 = CountVectorizer(binary=True)

        X = cvStep1.fit_transform(K)

        # Binarize the order to label whether an observation belongs to a client 1 or 0 a prospect
        y = y.copy()
        ybin = y
        ybin[ybin>0] = 1
        
        wordfreq = sum(X)
        wordfreq = wordfreq.toarray()[0,:]
        
        coef = np.array([])

        # Suppress print to avoid scipy returning "The exact solution is x=0"
        blockPrint()
        
        # Loop for every column in the matrix
        for i in X.T:
            coefficient = LinearRegression(fit_intercept=False).fit(i.T,ybin).coef_
            coef = np.concatenate((coef, coefficient))

        # Enable again printing 
        enablePrint()
        
        # Filter the coefficients based on the parameters
        coef[(coef < self.alpha_plus) * (coef > self.alpha_minus)] = np.nan
        coef[wordfreq < self.kappa] = np.nan

        self.marginal_screening = pd.DataFrame(({"term": cvStep1.get_feature_names(),
                                                 "score": coef
        })).dropna()


        # TOPIC MODELLING #
        # Create a column with only sentiment charged keywords
        K = K.apply(drop_non_sentiment_words, sentiment_words=self.marginal_screening["term"].to_list())

        # Remove entries without sentiment charged words
        y = y[K != ""]
        K = K[K != ""]
        
        # Define p-hat as the normalized rank
        y = y.rank(pct=True)

        # Initialize weight matrix 
        W = np.matrix([y, 1-y]).T

        # Compute count of sentiment charged words for each web-page
        s = K.apply(lambda row: len(row.split(" ")))

        # Create document keyword matrix
        cvStep2 = CountVectorizer()
        dS = cvStep2.fit_transform(K)

        # Get sentiment word frequency per document         
        tildeD = dS/s[:,None]

        # Fit the linear regression to estimate O matrix
        O = LinearRegression(fit_intercept=True).fit(X = W, y = tildeD).coef_

        # Set negative coefficients to 0
        O[O <= 0] = 0

        # Normalize result to l1
        normalize(O, norm='l1', axis=0, copy=False, return_norm=False)

        self.topic_coefficients = O

        return self


    def predict(self, K):
        """

        """
        def mle(x, s, dS, O):
            """ The function implements the log-likelihood of a multinomial with penalty as posed by Kelly et al.

            Parameters
            ----------
            x : float
                The sentiment score.
            s : int
                The number of sentiment charged words per web-page.
            dS : pandas series
                A series containing sentiment charged words frequencies.
            O : array-like
                Matrix containing word positiveness or negativeness.
                
            Returns
            -------
            v : float
                Return the log-likelihood value given x.

            """
            return -((float(s)**(-1)) *
                     np.sum(np.multiply(dS.toarray().T,(np.log(x*O[:,0] + (1-x)*O[:,1]))[:,None]) + self.l*np.log(x*(1-x))))

        # Create a column with only sentiment charged keywords
        K = K.apply(drop_non_sentiment_words, sentiment_words=self.marginal_screening["term"].to_list())

        # Compute count of sentiment charged words for the web-page
        s = K.apply(lambda row: len(row.split(" ")))

        # Create document keyword matrix
        cvStep3 = CountVectorizer(vocabulary=self.marginal_screening["term"].to_list())
        dS = cvStep3.fit_transform(K)

        # Get sentiment word frequency per document         
        D = dS/s[:,None]

        p = []
        for i in range(len(s)):
            p.append(fmin(mle, x0 = 0.01, args = (s.iloc[i], dS[i,:],self.topic_coefficients)))

        # Maximize the log-likelihood
        self.p = np.array(p)
        
        return self.p

