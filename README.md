# Assignment 4 - Using finetuned transformers via HuggingFace

## 1.	Contributions
Beyond troubleshooting errors with various guides, Stackoverflow pages, and ChatGPT, I was the only contributor to this assignment and the code itself was written by me, taking some notes from in-class work.

## 2.	Assignment Description
“In previous assignments, you've done a lot of model training of various kinds of complexity, such as training document classifiers or RNN language models. This assignment is more like Assignment 1, in that it's about feature extraction.
For this assignment, you should use ```HuggingFace``` to extract information from the Fake or Real News dataset that we've worked with previously.
You should write code and documentation which addresses the following tasks:
-	Initalize a HuggingFace pipeline for emotion classification
-	Perform emotion classification for every headline in the data
-	Assuming the most likely prediction is the correct label, create tables and visualisations which show the following: 
    - Distribution of emotions across all of the data
    - Distribution of emotions across only the real news
    - Distribution of emotions across only the fake news
-	Comparing the results, discuss if there are any key differences between the two sets of headlines”

## 3.	Methods
The data ```csv``` is loaded and separated according to real/fake label. The classifier is defined and applied in a ```for``` loop on all real- and fake-labeled headlines iteratively, taking the highest-probability emotion as the assumed correct one. The emotions identified are counted and categorized in a single dataframe containing the results from real, fake, and all headlines. This table is save to the ```out``` folder and used for further visualization work in a pair of bar plots, which are also saved to the ```out``` folder. 

## 4.	Usage
Firstly, install the required packages. For this, ensure that the current directory is ```assignment-4---using-finetuned-transformers-nikolaimh``` and run ```pip install –r requirements.txt``` from the terminal. Afterwards, execute ```cd src/``` and run the main scripts by executing ```python pipeline.py``` from the terminal. The data used is already present in the ```data``` folder and the output plots will be saved to the ```out``` folder.
 
## 5.	Discussion
The full table of results is as follows:

|        |All headlines|Real headlines|Fake headlines|
|--------|-------------|--------------|--------------|
|Anger   |795          |383           |412           |
|Disgust |434          |186           |248           |
|Fear    |1076         |555           |521           |
|Joy     |155          |63            |92            |
|Neutral |3180         |1649          |1531          |
|Sadness |487          |245           |242           |
|Surprise|208          |90            |118           |

Perhaps unsurprisingly, the first thing one might not is the preponderance of articles headlines themed to neutrality; across the real, fake, and mixed category, most articles are neutral in emotional tone, though this is followed by fear and anger in all cases, with disgust and sadness sharing a roughly equal fourth. Barely any article headlines seem to emphasize joy or surprise, the latter of which I would expect to be markedly higher with data from some outlets over others. The bar plot below shows this distribution:

![image](https://github.com/AU-CDS/assignment-4---using-finetuned-transformers-nikolaimh/assets/112465764/e93e72ed-6b62-4a23-b7d3-64ced84771e5)
 
As for the results concerning the real- and fake-labeled headlines, the difference is far less striking than I had initially expected. I would have hypothesized that neutrality would be far higher among real headlines, while the fake ones would emphasize negative emotions; anger, fear, and disgust in particular. This does not appear to be so, as illustrated by the comparison plot:

![image](https://github.com/AU-CDS/assignment-4---using-finetuned-transformers-nikolaimh/assets/112465764/c6428397-6081-48bc-8cad-2e5c7c2e10c6)
 
Here, the difference seems rather insignificant. None of the categories seem so cleanly separated as to distinguish any marked variance. Whether this data is representative and the pattern holds true for headlines sourced from other places and outlets remains uncertain, but the emotional tone seems consistent regardless of an article’s veracity.
