import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sbs
import plotly.graph_objects as go
import plotly.express as pexp

from collections import Counter



class Analyzer:
    def __init__(self, dataset):
        dataset['Year'] = dataset['Year'].fillna(0)
        dataset['Release_year'] = dataset['Year'].apply(lambda y: str(y).split("–")[0]).replace('\D+', '', regex=True)

        dataset['Rating'] = dataset['Rating'].str.replace("-", "0")

        dataset = dataset.rename(columns={'Votes:': 'Vote'})  # Rename the Vote column, so that it's easier to work with
        dataset['Vote'] = dataset['Vote'].str.replace("-", "0")
        dataset['Vote'] = dataset['Vote'].str.replace(",", "")

        dataset['Time'] = dataset['Time'].apply(lambda y: str(y).replace("-", "0").replace("min", "").strip())

        dataset['Short_Story'] = dataset['Short_Story'].str.replace("\n", "")

        for col in ['Release_year', 'Rating', 'Vote', 'Time']:
            dataset[col] = dataset[col].apply(pd.to_numeric)

        dataset.drop('Year', axis=1, inplace=True)
        self.dataset = dataset

    #Release Year Analysis
    def getYearChart(self):
        plt.figure(figsize= (12,4))
        sbs.countplot(x = 'Release_year', data = self.dataset[self.dataset['Release_year'] != 0], palette = 'pastel')
        plt.xticks(rotation = 45)
        print(plt.show())

    #Genre Analysis
    def getGenreChartAnalysis(self):
        genreList = list()
        for genre in self.dataset['Genre'].to_list():
            for g in genre.split(", "):
                if g != "-":
                    genreList.append(g)
        genreDataFrame = pd.DataFrame.from_dict(Counter(genreList), orient = 'index', columns = ['Frequency']).reset_index()
        genreDataFrame = genreDataFrame.rename(columns = {'index' : 'Category'})
        genreDataFrame = genreDataFrame.sort_values('Frequency', ascending = False)
        return genreDataFrame


    def getGenreGraph(self, dataframe):
        figure = pexp.bar(dataframe, x = 'Category', y = 'Frequency', title = 'Korean TV-Series Category vs Frequency')
        return figure

    #Cast Stars Analysis
    def getStarsChartAnalysis(self):
        nonAnimatedShows = self.dataset[self.dataset['Genre'].str.contains("Animation") == False]
        starsList = []

        for stars in nonAnimatedShows['Stars'].to_list():
            star = stars.split(", ")
            for s in star:
                if s != "-":
                    starsList.append(s)
        starDataFrame = pd.DataFrame.from_dict(Counter(starsList), orient = 'index').reset_index()
        starDataFrame = starDataFrame.rename(columns = {"index": "Star", 0:'Frequency'}).sort_values(by = 'Frequency', ascending = False)
        return starDataFrame

    #Main Cast vs Rating Chart
    def getStarVSRating(self, starDataFrame):
        cutoff = starDataFrame['Frequency'].quantile(0.99)
        top1Star = starDataFrame[starDataFrame['Frequency'] > cutoff]['Star'].tolist()

        for star in top1Star:
            temp = self.dataset[self.dataset['Stars'].str.contains(star)]
            avgRating = temp['Rating'].mean()
            print(star, ":", round(avgRating, 3))

    #Get Top 10 Shows
    def getTopTen(self):
        print(self.dataset.sort_values(by = 'Rating', ascending = False).head(10))

