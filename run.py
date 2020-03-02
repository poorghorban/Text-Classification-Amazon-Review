'''
******************** TEXT CLASSIFICATION AMAZON REVIEW **********************
article link:
https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/

requirement:
- python version 3.x
- pip install pandas 
- pip install seaborn
- pip install matplotlib
- pip install nltk
- pip install sklearn
'''
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import nltk
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

################################### FUNCTIONS ###################################

## cleaned text review 
# 1) convert to lowercase
# 2) remove punctuation
# 3) remove digit
# 4) tokenize
# 5) remove stopwords 
# 6) lemmatize (word rooting)
def cleaned_text(text):

    # step1: convert to lowercase
    text = text.lower()

    # step2 & step3: remove punctuation & remove digit
    str_digit = '0123456789'
    no_punc_no_digit_text = ''.join([c for c in text if (c not in punctuation + str_digit)])
    
    # step4: tokenize
    tokenize_word = nltk.word_tokenize(no_punc_no_digit_text)

    # step5 & step6: remove stopwords & stemer
    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = nltk.stem.WordNetLemmatizer()

    no_stopwrods_lemmatize_words = [lemmatizer.lemmatize(w) for w in tokenize_word if w not in stopwords]

    return no_stopwrods_lemmatize_words



################################### MAIN ###################################

if __name__ == "__main__":
    
    ################################### Step 1: Input Data ###################################

    ##### read data from file to path data/corpus and save to list by name text and label #####
    ## file path 
    path = "data\corpus"
    
    ## open file 
    f = open(file=path , mode='r' , encoding="utf-8")

    ## list text for save all review amazon 
    text = []

    ## list label for save label all review 
    label = []

    ## read line by line file 
    ## structure line : __label__'number label' 'text review'
    for line in f:

        # split section label and text with first space
        split_line = line.split(" " , 1)

        # save text to list text 
        text.append(split_line[1])

        # save label to list label 
        # remove '__label__' and get number 
        number_label = split_line[0].split("__label__")
        label.append(number_label[1])

    ##### save lists in Panda DataFrame for ease of handling #####
    # both lists, with columns specified 
    df = pd.DataFrame(list(zip(text , label)) , columns=['text' , 'label'])

    ################################### Step 2: Exploratory Data Analysis ###################################

    ##### univariate anaylsis #####

    ## show barplot and boxplot 
    fig, axs = plt.subplots(ncols=2)
    # set title figure 
    fig.suptitle('Exploratory Data Analysis', fontsize=16)
    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette("bright"))

    ## show barplot of the number of review for label groups
    # get number of review for each category label
    df_barplot = pd.DataFrame(df['label'].value_counts())
    df_barplot = df_barplot.reset_index()
    df_barplot.columns = ['label', 'number of text']

    # set barplot 
    axs[0].set_title('BarPlot')
    sns.barplot(x="label", y="number of text", data=df_barplot , ax=axs[0])

    ## show boxplot of length of each text review in label groups
    # get length of each text review and add dataframe label
    df_boxplot = (df.text.str.len()).to_frame()
    df_boxplot.columns = ['length of text']
    df_boxplot['label'] = df['label']

    # set boxplot
    axs[1].set_title('BoxPlot')
    sns.boxplot(x="label", y="length of text", data=df_boxplot , ax=axs[1])

    # show with matplotlib
    plt.show()

    ################################### Step 3: Feature Engineering ###################################

    ##### Cleaend text and convert words into vectors of binary numbers #####
    ##### convert words into vectors of binary numbers methods: 1)countvectorizer 2)tfidfvectorizer

    ## first method : Count Vectorizer
    countVectorizer = CountVectorizer(analyzer=cleaned_text , max_features=5000 , encoding="utf-8")
    X_countVectorizer = countVectorizer.fit_transform(df['text']).toarray()


    ## second method : TF_IDF Vectorizer 
    tfidfVectorizer = TfidfVectorizer(analyzer=cleaned_text , max_features=5000 , encoding="utf-8")
    X_tfidfVectorizer = tfidfVectorizer.fit_transform(df['text']).toarray()

    ################################### Step 4: Traning Model ###################################

    ##### train model for data count vectorizer #####

    # split data countvectorizer to train and valid 
    x_train_countvectorizer , x_valid_countvectorizer , y_train_countvectorizer , y_valid_countvectorizer = train_test_split(X_countVectorizer, df['label'] ,  test_size=0.2 , random_state=144 , shuffle=True)

    # train model 
    logisticRegressionModel = LogisticRegression(solver='lbfgs' , max_iter=10000)
    logisticRegressionModel.fit(x_train_countvectorizer , y_train_countvectorizer)

    # predict data valid 
    y_valid_predict_countvectorizer = logisticRegressionModel.predict(x_valid_countvectorizer)

    # get classification report 
    classification_report_countvectorizer = classification_report(y_valid_countvectorizer , y_valid_predict_countvectorizer , target_names=['class1' , 'class2'])
    
    ##### train model for data tfidf vectorizer #####

    # split data countvectorizer to train and valid 
    x_train_tfidfvectorizer , x_valid_tfidfvectorizer , y_train_tfidfvectorizer , y_valid_tfidfvectorizer = train_test_split(X_tfidfVectorizer, df['label'] ,  test_size=0.2 , random_state=144 , shuffle=True)

    # train model 
    logisticRegressionModel.fit(x_train_tfidfvectorizer , y_train_tfidfvectorizer)

    # predict data valid 
    y_valid_predict_tfidfvectorizer = logisticRegressionModel.predict(x_valid_tfidfvectorizer)

    # get classification report 
    classification_report_tfidfvectorizer = classification_report(y_valid_tfidfvectorizer , y_valid_predict_tfidfvectorizer  , target_names=['class1' , 'class2'])

    ################################### Step 5: Show Result ###################################
 
    ##### show confusion matrix  #####

    fig, axs = plt.subplots(ncols=2)
    
    # remove grid 
    axs[0].grid(False)
    axs[1].grid(False)

    # set title figure 
    fig.suptitle('LogisticRegression Model(compare confusion matrix)', fontsize=16)
    
    # show confusion matrix countvectorizer 
    disp = plot_confusion_matrix(logisticRegressionModel, x_valid_countvectorizer, y_valid_countvectorizer,
                                 display_labels=['class1' , 'class2'],
                                 cmap=plt.cm.Blues,
                                 normalize='true' , ax=axs[0])
    disp.ax_.set_title("CountVectorizer")

    # show confusion matrix countvectorizer 
    disp1 = plot_confusion_matrix(logisticRegressionModel, x_valid_tfidfvectorizer, y_valid_tfidfvectorizer,
                                 display_labels=['class1' , 'class2'],
                                 cmap=plt.cm.Blues,
                                 normalize='true' , ax=axs[1])
    disp1.ax_.set_title("TfIdfVectorizer")

    # show plot 
    plt.show()

    ##### show classification report #####
    print("Model: LogisticRegression")

    ## count vectorizer
    print("First Method: CountVectorizer")
    print(classification_report_countvectorizer)

    ## tfidf vectorizer
    print("Second Method: TfIdfVectorizer")
    print(classification_report_tfidfvectorizer)

    ################################### Step 6: Dimensionality Reduction ###################################
    '''
    ##### PCA #####
    pca = PCA(n_components=2)
    x_pca_countvectorizer = pca.fit_transform(pd.DataFrame(X_countVectorizer))

    df_pca_countvectorizer = pd.DataFrame(x_pca_countvectorizer , columns=['feature1' , 'feature2'])
    df_pca_countvectorizer['label'] = df['label']

    x_pca_tfidfvectorizer = pca.fit_transform(pd.DataFrame(X_tfidfVectorizer))
    df_pca_tfidfvectorizer = pd.DataFrame(x_pca_tfidfvectorizer , columns=['feature1' , 'feature2'])
    df_pca_tfidfvectorizer['label'] = df['label']


    ##### TSNE #####
    tsne = TSNE(n_components=2)
    x_tsne_countvectorizer = tsne.fit_transform(pd.DataFrame(X_countVectorizer))
    df_tsne_countvectorizer = pd.DataFrame(x_tsne_countvectorizer , columns=['feature1' , 'feature2'])
    df_tsne_countvectorizer['label'] = df['label']

    x_tsne_tfidfvectorizer = tsne.fit_transform(pd.DataFrame(X_tfidfVectorizer))
    df_tsne_tfidfvectorizer = pd.DataFrame(x_tsne_tfidfvectorizer , columns=['feature1' , 'feature2'])
    df_tsne_tfidfvectorizer['label'] = df['label']

    ##### Show plot pca and tsne for countvectorizer and tfidfvectorizer

    fig, axs = plt.subplots(nrows=2, ncols=2)
    # set title figure 
    fig.suptitle('Dimensionality Reduction(PCA - TSNE)', fontsize=16)
    sns.set(style="whitegrid")
    #sns.set_palette(sns.color_palette("bright"))

    # plot pca- countvectorizer 
    axs[0,0].set_title("pca-countvectorizer")
    sns.lmplot(x = 'feature1' , y='feature2' , data=df_pca_countvectorizer , fit_reg=False, hue='label',scatter_kws={'s':100 , 'alpha':0.1} , ax=axs[0,0])

    # plot tsne- tfidfvectorizer
    axs[0,1].set_title("tsne-countvectorizer")
    sns.lmplot(x = 'feature1' , y='feature2' , data=df_tsne_countvectorizer , fit_reg=False, hue='label',scatter_kws={'s':100 , 'alpha':0.1} , ax=axs[0,1])

    # plot pca- countvectorizer 
    axs[1,0].set_title("pca-tfidfvectorizer")
    sns.lmplot(x = 'feature1' , y='feature2' , data=df_pca_tfidfvectorizer , fit_reg=False, hue='label',scatter_kws={'s':100 , 'alpha':0.1} , ax=axs[1,0])

    # plot tsne- tfidfvectorizer
    axs[1,1].set_title("tsne-tfidfvectorizer")
    sns.lmplot(x = 'feature1' , y='feature2' , data=df_tsne_tfidfvectorizer , fit_reg=False, hue='label',scatter_kws={'s':100 , 'alpha':0.1} , ax=axs[1,1])

    # show plot 
    plt.show()
    '''