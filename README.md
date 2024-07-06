Dataset Description:

The dataset relates to legal decisions of Nigerian Courts, consisting of 500 rows and 6 columns. The columns include CASE TITLE, SUIT NO, CITATION, STORY, ISSUES, and an S/N column mostly populated with NaN values.
•	CASE TITLE: Contains names of parties involved in each case (formatted as "Plaintiff v. Defendant" or "Appellant v. Respondent"). It features 480 unique entries with no missing values.
•	SUIT NO: Serves as a unique identifier for each case, with all 500 entries being unique and free of missing values.
•	CITATION: Provides specific references for each case, generally formatted as "(Year) LPELR-XXXXX(CA)." All 500 citations are unique and contain no missing values.
•	STORY: Includes detailed narratives or introductions and Facts about each case, providing background information and context. Issues and Decisions held by the Appeal court for each of the cases. There are 499 unique stories with one missing value.
•	ISSUES: Lists the key legal issues or points of law addressed in each case, with 498 unique entries and no missing values.
The attributes and their data types indicate that most data is textual, suitable for Natural Language Processing (NLP) applications to extract insights and predict legal outcomes. The dataset is well-organized and detailed, making it suitable for legal research, predictive modeling, and text analysis.
This dataset is sourced from a pool originally hosted by Law Pavillion Electronic Law Report, a Nigerian-registered and operating online law reporting company. Link: https://docs.google.com/spreadsheets/d/1Nc5pnhB7saXPZBUJr6OuZ-sXlpdwe-aKEyqSMSUj-Ns/edit?usp=drivesdk
[Prediction of Legal Decisions using Machine Learning (1).xlsx](https://github.com/user-attachments/files/15807173/Prediction.of.Legal.Decisions.using.Machine.Learning.1.xlsx)

 
Steps:

Data Loading and Initial Exploration:
•	The code begins by importing necessary libraries (pandas, re, nltk, stopwords, WordNetLemmatizer, TfidfVectorizer, matplotlib.pyplot, seaborn, WordCloud, tensorflow, Sequential, Dense, Dropout) for data handling, text preprocessing, visualization, machine learning, and deep learning.
•	It loads a dataset named dataset.xlsx using pd.read_excel into a Pandas DataFrame data.
•	The first few rows of the dataset are displayed using data.head().

Exploratory Data Analysis (EDA):
•	It uses Seaborn to create a count plot (sns.countplot) to visualize the distribution of 'SUIT NO' in the dataset.

Feature Engineering:
•	Extracts features from 'CASE TITLE' by splitting it into 'PLAINTIFF' and 'DEFENDANT' using str.split and expands them into separate columns in data.
•	Handles missing values in 'PLAINTIFF' and 'DEFENDANT' by filling them with 'Unknown'.
•	Creates a new feature 'CASE_TYPE' based on whether 'CASE TITLE' contains ' v. ' indicating a civil case or not.

Text Preprocessing:
•	Downloads necessary NLTK resources (punkt, wordnet, stopwords).
•	Defines a function preprocess_text to clean and tokenize text:
o	Removes non-alphabetic characters using regular expressions (re.sub).
o	Converts text to lowercase.
o	Tokenizes text into words using NLTK's word_tokenize.
o	Lemmatizes words and removes stopwords.
•	Applies preprocess_text to 'STORY' and 'ISSUES' columns, storing the cleaned text in new columns 'Cleaned_STORY' and 'Cleaned_ISSUES' in data.

Visualization:
•	Creates a count plot (sns.countplot) to visualize the distribution of 'CASE_TYPE' (civil vs unknown).
•	Generates a word cloud (WordCloud) for 'Cleaned_STORY' to visually represent the most frequent words in the text data.

Textual Features: TF-IDF Vectors:
•	Initializes two TfidfVectorizer objects (tfidf_vectorizer_story and tfidf_vectorizer_issues) to convert 'Cleaned_STORY' and 'Cleaned_ISSUES' into TF-IDF matrices (tfidf_story and tfidf_issues).

Label Encoding:
•	Imports LabelEncoder from Scikit-learn.
•	Encodes categorical target variable y using label_encoder.fit_transform to convert categorical labels into numerical indices (y_encoded).

Train-Test Split:
•	Splits the data (X and y_encoded) into training and testing sets (X_train, X_test, y_train, y_test) using train_test_split.

Neural Network Model:
•	Defines a neural network model using Sequential from Keras.
•	Adds layers (Dense) with ReLU activation and Dropout for regularization.
•	Compiles the model with sparse_categorical_crossentropy as the loss function (appropriate for integer-encoded labels) and adam optimizer.
•	Trains the model (model.fit) on training data (X_train, y_train) with validation on testing data (X_test, y_test) over 10 epochs.

Predictions and Evaluation:
•	Makes predictions (y_pred_probs) on test data using model.predict.
•	Converts predicted probabilities to class labels (y_pred) by selecting the index with the highest probability.
•	Prints a classification report (classification_report) showing precision, recall, F1-score, and support for each class based on true (y_test) and predicted (y_pred) labels.
![image](https://github.com/ChristopherEban/Prediction-of-Legal-Decisions-Using-ML-Dataset-of-Nigeria-Appeal-Court-Case-Authorities-/assets/164779510/b54f6896-b867-4627-8a81-5293428a7c2b)


