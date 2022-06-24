# Author: Leo Li
# Purpose: Created a Korean TV Show Menu that conducts Data Analysis of
# an IMDb Korean TV Series Dataset.
# Also contains a content-based tv show recommendation system that utilizes natural
# language processing (NLP)

#Reference:
#https://towardsdatascience.com/how-to-build-from-scratch-a-content-based-movie-recommender-with-natural-language-processing-25ad400eb243

#Dataset:
#https://www.kaggle.com/datasets/chanoncharuchinda/imdb-korean-tv-series

#Importing Libraries
import pandas as pd

from IPython.display import display
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from ShowAnalyzer import Analyzer

def getRawData():
    shows = pd.read_csv("koreanTV.csv")
    return shows

def getFilteredData():
    kDramas = pd.read_csv("koreanTV.csv")
    kDramas = kDramas[['Title', 'Genre', 'Stars', 'Short_Story']]
    return kDramas

def cleanData(dataFrame):
    #Removing commas between stars' full names and getting only the first three stars' names
    dataFrame['Stars'] = dataFrame['Stars'].map(lambda x: x.split(',')[:3])

    #Putting Listed Genres into a list of words
    dataFrame['Genre'] = dataFrame['Genre'].map(lambda x: x.lower().split(','))

    #Extraction of key words from Short Story (Synopsis) Description
    dataFrame['key_words'] = ""

    #Removing '\n' characters in 'Short Story' Column
    dataFrame['Short_Story'] = dataFrame['Short_Story'].str.replace("\n", "")

    for index, row in dataFrame.iterrows():
        synopsis = row['Short_Story']

        #Instantiating Rake: uses English stopwords from NLTK and discards all punctuation characters
        r = Rake()

        #Extracting key words by passing in a synopsis row from data set
        r.extract_keywords_from_text(synopsis)

        #Getting the dictionary with key words and corresponding scores
        keywords_dict_scores = r.get_word_degrees()

        #Assigning the extracted key words to the new column declared before for loop
        row['key_words'] = list(keywords_dict_scores.keys())

    #Drop 'Short Story' column
    dataFrame.drop(columns = ['Short_Story'], inplace = True) #inplace True meaning not returning anything

    dataFrame.set_index('Title', inplace = True)

    dataFrame['bag_of_words'] = ''
    columns = dataFrame.columns
    for index, row in dataFrame.iterrows():
        words = ''
        for col in columns:
            words = words + ' '.join(row[col]) + ' '
        row['bag_of_words'] = words

    dataFrame.drop(columns = [col for col in dataFrame.columns if col != 'bag_of_words'], inplace = True)
    return dataFrame

def cosineSimilarityCalc(dataFrame):
    #Instantiating and generating the a matrix that stores the frequency for each word in 'bag_of_words' column
    wordCount = CountVectorizer()
    countMatrix = wordCount.fit_transform(dataFrame['bag_of_words'])

    #Generate Cosine Similarity Matrix
    cosineSim = cosine_similarity(countMatrix, countMatrix)
    return cosineSim

#Function that takes in a show title as input and returns the top 10 recommended movies
def getShowRecommendation(dataFrame, title, cosineSim, indices):
    recommendedShows = []

    #Get the index of the show that matches the title
    index = indices[indices == title].index[0]

    #Create a series with the similarity scores in descending order
    scoreSeries = pd.Series(cosineSim[index]).sort_values(ascending = False)

    #Get the index of the top 10 most similar tv shows
    top10Indexes = list(scoreSeries.iloc[1:11].index)

    #Populate the list with titles of the top 10 most similar tv shows
    for index in top10Indexes:
        recommendedShows.append(list(dataFrame.index)[index])

    return recommendedShows

def inDataSet(kShowsDataSet, showTitle):
    for i in range(len(kShowsDataSet.Title)):
        if showTitle == kShowsDataSet.Title[i]:
            return True
    return False

def getShowSynopses(kShowsDataSet, showList):
    #Steps:
    # 1. Search for index where the show appears in the Titles column
    # 2. Access the Short Story value using that found index
    # 3. Store that String into synopsisList
    # 4. Repeat for rest of shows
    # 5. Return the synopsisList

    synopsisList = []
    #Accessing synopsis column: kShowDataSet.iloc[:3]
    for i in range(len(showList)):
        for j in range(len(kShowsDataSet.Title)):
            if showList[i] == kShowsDataSet.Title[j]:
                synopsisList.append(kShowsDataSet.Short_Story[j])
    return synopsisList

def getSingleShowSynopsis(kShowsDataSet, showTitle):
    for i in range(len(kShowsDataSet.Title)):
        if showTitle == kShowsDataSet.Title[i]:
            return kShowsDataSet.Short_Story[i]

def displayMenu():
    print("Welcome to Korean TV Show Recommender/Analyzer!")
    print("Enter one of the following numbers for options:")
    print("1. Search for Specific Show and its Information")
    print("2. Get TV Show Recommendation")
    print("3. Get TV Show Year Graph")
    print("4. Get TV Genre Chart Showing Category and Frequency")
    print("5. Get Genre vs Frequency Bar Graph")
    print("6. Get Actor Chart Showing Star and Frequency")
    print("7. Get Actor Chart Showing Star and Show Rating")
    print("8. Get Top 10 Highest Rated Shows")
    print("9. Quit the application.\n")

def main():
    origDataSet = getFilteredData()

    while True:
        displayMenu()

        invalidChoice = True
        choice = int(input(""))

        #Validating user choice is between 1 and 9 (inclusive)
        while invalidChoice:
            if choice < 1 or choice > 9:
                choice = input("Invalid Option. Try again.")
                displayMenu()
            else:
                invalidChoice = False

        #Search for Specific Show and its Information
        if choice == 1:
            print("Enter a K-Drama Title according to it's name in IMDb: ")
            kDramaTitle = input("")

            # Gathering Dataset
            kShows = getFilteredData()

            # Validating Show Title
            while not inDataSet(kShows, kDramaTitle):
                print("Title Not In IMDb Data Set. Try Entering Another Title: ")
                kDramaTitle = input("")

            showSynopsis = getSingleShowSynopsis(kShows, kDramaTitle)

            print(f'{kDramaTitle}')
            print(f'Synopsis: {showSynopsis}\n')

        #Get TV Show Recommendation
        elif choice == 2:
            print("Enter a K-Drama Title according to it's name in IMDb: ")
            kDramaTitle = input("")

            #Gathering Dataset
            kShows = getFilteredData()

            #Validating Show Title
            while not inDataSet(kShows, kDramaTitle):
                print("Title Not In IMDb Data Set. Try Entering Another Title: ")
                kDramaTitle = input("")

            #Cleaning the Data
            kShows = cleanData(kShows)

            similarityMatrix = cosineSimilarityCalc(kShows)

            # Create Series for show titles so that they are associated to an ordered numerical list
            # Used to match the indexes from the similarity matrix to the actual show titles
            indices = pd.Series(kShows.index)

            #Formatting Recommendation Results
            resultList = getShowRecommendation(kShows, kDramaTitle, similarityMatrix, indices)
            synopList = getShowSynopses(origDataSet, resultList)
            showNum = 1
            for index in range(len(resultList)):
                print(f'{showNum}. {resultList[index]}')
                print(f'Synopsis: {synopList[index]}\n')
                showNum += 1

        #Get TV Show Year Graph
        elif choice == 3:
            rawDataSet = getRawData()
            showAnalyzer = Analyzer(rawDataSet)
            showAnalyzer.getYearChart()
            print()

        #Get TV Genre Chart Showing Category and Frequency
        elif choice == 4:
            rawDataSet = getRawData()
            showAnalyzer = Analyzer(rawDataSet)
            genreDataFrame = showAnalyzer.getGenreChartAnalysis()
            print(display(genreDataFrame))
            print()

        #Get Genre vs Frequency Bar Graph
        elif choice == 5:
            rawDataSet = getRawData()
            showAnalyzer = Analyzer(rawDataSet)
            genreDataFrame = showAnalyzer.getGenreChartAnalysis()
            barGraph = showAnalyzer.getGenreGraph(genreDataFrame)
            print(barGraph.show())
            print()

        #Get Actor Chart Showing Star and Frequency
        elif choice == 6:
            rawDataSet = getRawData()
            showAnalyzer = Analyzer(rawDataSet)
            starDataFrame = showAnalyzer.getStarsChartAnalysis()
            print(display(starDataFrame))
            print()

        #Get Actor Chart Showing Star and Show Rating
        elif choice == 7:
            rawDataSet = getRawData()
            showAnalyzer = Analyzer(rawDataSet)
            starDataFrame = showAnalyzer.getStarsChartAnalysis()
            showAnalyzer.getStarVSRating(starDataFrame)
            print()

        #Get Top 10 Shows
        elif choice == 8:
            rawDataSet = getRawData()
            showAnalyzer = Analyzer(rawDataSet)
            print(showAnalyzer.getTopTen())
            print()

        #Quit Application
        elif choice == 9:
            print("Goodbye")
            break


if __name__ == "__main__":
    main()








