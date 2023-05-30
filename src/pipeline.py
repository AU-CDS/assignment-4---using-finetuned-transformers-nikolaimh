# pathing tool
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # minimize tensorflow messages
# dataframe wrangling
import pandas as pd
# classifier used
from transformers import pipeline
###import tensorflow
# for visualisations
from matplotlib import pyplot as plt

# loading .csv data and separating into lists of real and fake headlines
def load_data():
    # reading input data
    data_path = os.path.join("..","data","fake_or_real_news.csv")
    realfake_df = pd.read_csv(data_path)

    headlines = list(realfake_df["title"])
    real_headlines = []
    fake_headlines = []

    for idx, row in realfake_df.iterrows():
        if row["label"] == "REAL":
            real_headlines.append(row["title"])
        elif row["label"] == "FAKE":
            fake_headlines.append(row["title"])
        else:
            pass

    return headlines, real_headlines, fake_headlines

# define emotion classifier
def set_cls():
    classifier = pipeline("text-classification", 
                          model="j-hartmann/emotion-english-distilroberta-base", 
                          return_all_scores=True)
    return classifier

# run data through classifier, assuming highest prob emotion as correct
def analyse_data(classifier, headlines, real_headlines, fake_headlines):    
    real_verdicts = []

    print("     classifying real headlines ...")
    for headline in real_headlines:
        verdict = classifier(headline)
        # unlisting verdict by one level
        verdict = verdict[0]
        # finding dict with highest probability
        max_score = max(verdict, key=lambda x:x["score"])
        # getting label from highest prob dict
        best_label = max_score["label"]
        # adding to total list
        real_verdicts.append(best_label)

    fake_verdicts = []

    print("     classifying fake headlines ...")
    for headline in fake_headlines:
        verdict = classifier(headline)
        verdict = verdict[0]
        max_score = max(verdict, key=lambda x:x["score"])
        best_label = max_score["label"]
        fake_verdicts.append(best_label)

    headline_verdicts = real_verdicts + fake_verdicts

    return headline_verdicts, real_verdicts, fake_verdicts

# creating tables from counts of each emotion in verdicts
def make_table(headline_verdicts, real_verdicts, fake_verdicts):
    # counting highest probability emotions in headlines
    all_anger = headline_verdicts.count("anger")
    all_disgust = headline_verdicts.count("disgust")
    all_fear = headline_verdicts.count("fear")
    all_joy = headline_verdicts.count("joy")
    all_neutral = headline_verdicts.count("neutral")
    all_sadness = headline_verdicts.count("sadness")
    all_surprise = headline_verdicts.count("surprise")

    real_anger = real_verdicts.count("anger")
    real_disgust = real_verdicts.count("disgust")
    real_fear = real_verdicts.count("fear")
    real_joy = real_verdicts.count("joy")
    real_neutral = real_verdicts.count("neutral")
    real_sadness = real_verdicts.count("sadness")
    real_surprise = real_verdicts.count("surprise")

    fake_anger = fake_verdicts.count("anger")
    fake_disgust = fake_verdicts.count("disgust")
    fake_fear = fake_verdicts.count("fear")
    fake_joy = fake_verdicts.count("joy")
    fake_neutral = fake_verdicts.count("neutral")
    fake_sadness = fake_verdicts.count("sadness")
    fake_surprise = fake_verdicts.count("surprise")

    # creating dataframe with all results
    emotion_table = pd.DataFrame({"Anger": [all_anger, real_anger, fake_anger],
                                  "Disgust": [all_disgust, real_disgust, fake_disgust],
                                  "Fear": [all_fear, real_fear, fake_fear],
                                  "Joy": [all_joy, real_joy, fake_joy],
                                  "Neutral": [all_neutral, real_neutral, fake_neutral],
                                  "Sadness": [all_sadness, real_sadness, fake_sadness],
                                  "Surprise": [all_surprise, real_surprise, fake_surprise]})

    # transposing df and naming columns for better display
    new_table = emotion_table.T
    new_table = new_table.rename(columns={0:"All headlines",
                                          1:"Real headlines",
                                          2:"Fake headlines"})

    return new_table, emotion_table

def visualise_results(emotion_table):
    # separating emotions from df by headline type
    all_emotions = emotion_table.iloc[0]
    real_emotions = emotion_table.iloc[1]
    fake_emotions = emotion_table.iloc[2]

    # creating bar plot for general headline overview
    all_df = pd.DataFrame({"All headlines": all_emotions})
    all_vis = all_df.plot.bar()
    plt.title("Emotion distribution in all headlines")
    plt.xticks(rotation=45, ha="right")
    plt.xticks(fontsize=8)
    
    # creating bar plot for real/fake comparison
    comparison_df = pd.DataFrame({"Real headlines": real_emotions,
                                  "Fake headlines": fake_emotions})
    real_fake_vis = comparison_df.plot.bar()
    plt.title("Emotion distribution in real and fake headlines")
    plt.xticks(rotation=45, ha="right")
    plt.xticks(fontsize=8)
    return all_vis, real_fake_vis

def save_func(new_table, all_vis, real_fake_vis):
    # defining path to out folder for all results
    out_dir = os.path.join("..","out")

    # saving table
    table_path = os.path.join(out_dir,"emotion_table.csv")
    new_table.to_csv(table_path)

    # saving bar plot for all headline emotions
    all_plot_path = os.path.join(out_dir,"all_plot.png")
    all_vis.get_figure().savefig(all_plot_path)
    # saving bar plot for real/fake headline emotions
    real_fake_path = os.path.join(out_dir,"real_fake_plot.png")
    real_fake_vis.get_figure().savefig(real_fake_path)
    return None

def main():
    # load all data and sort according to real/fake label
    print("Loading data ...")
    all_hd, real_hd, fake_hd = load_data()
    # run headlines through defined classifier, outputting most likely emotions for each string
    emotion_cls = set_cls()
    print("   ")
    print("Processing verdicts:")
    headline_verdicts, real_verdicts, fake_verdicts = analyse_data(emotion_cls, all_hd, real_hd, fake_hd)

    # count emotion instances and make tables from results
    print("Visualising results ...")
    final_table, vis_table = make_table(headline_verdicts, real_verdicts, fake_verdicts)
    # make a bar plot of the results
    all_plot, real_fake_plot = visualise_results(vis_table)
    # save output table and plots to out folder
    save_func(final_table, all_plot, real_fake_plot)
    print("Visualisations saved to the out folder.")
    return None

if __name__ == "__main__":
    main()