# BBC News Topic Modeling

This project applies **Topic Modeling** techniques to uncover hidden themes in the **BBC News Dataset**. By combining advanced text preprocessing, vectorization methods, and unsupervised learning models, the goal is to extract meaningful topics from raw text and compare the performance of **Latent Dirichlet Allocation (LDA)** and **Non-negative Matrix Factorization (NMF)**.

---

## Dataset

We use the [**BBC News Dataset**](https://www.kaggle.com/datasets/gpreda/bbc-news) from Kaggle.

* Contains **2,225 news articles**
* Belongs to **5 categories**: *business, entertainment, politics, sport, tech*
* Each record includes:

  * **Category** (label)
  * **Text** (news article content)

This dataset is suitable for both **supervised text classification** and **unsupervised topic modeling**. In this project, we focus on the **unsupervised** side.

---

## Project Workflow

The notebook is structured as follows:

### 1. **Data Loading & Exploration**

* Load the BBC dataset into a pandas DataFrame.
* Inspect shape, sample rows, and category distribution.
* Plot category frequency to understand class balance.

### 2. **Text Preprocessing**

Performed to clean and normalize text before vectorization:

* Lowercasing
* Removing punctuation, digits, and special characters
* Tokenization with **NLTK**
* Stopword removal (English stopwords from `nltk.corpus`)
* Lemmatization using **WordNetLemmatizer**

### 3. **Feature Extraction**

Two representations were tested:

* **Bag-of-Words (CountVectorizer)** – basic word occurrence counts
* **TF-IDF (TfidfVectorizer)** – weighting terms by importance across documents

Both representations were experimented with n-grams and max feature limits to optimize performance.

### 4. **Latent Dirichlet Allocation (LDA)**

* LDA model was trained using **Gensim** and **scikit-learn’s LDA**.
* Hyperparameters tuned:

  * `n_components` (number of topics)
  * `max_iter` (iterations)
  * `learning_decay` (update rate)
* Extracted top keywords per topic.
* Visualized using **pyLDAvis** for interpretability.

### 5. **Non-negative Matrix Factorization (NMF)**

* Applied NMF on **TF-IDF vectors**.
* Compared topic coherence and interpretability with LDA.
* Displayed top words per topic for evaluation.

### 6. **Evaluation & Comparison**

* Both models were analyzed qualitatively:

  * LDA: good probabilistic modeling of topics, interpretable via pyLDAvis.
  * NMF: often produced sharper and more distinct topics.
* Final comparison highlighted trade-offs between LDA and NMF.

---

## Results

* **LDA** captured broader semantic structures and worked well with both BoW and TF-IDF.
* **NMF** provided more **distinct and interpretable topics**, especially with TF-IDF.
* **Visualization with pyLDAvis** allowed for clear inspection of topic separation and top keywords.

Sample extracted topics included:

* **Business** → "company, market, shares, growth, investors"
* **Sport** → "team, match, game, win, player"
* **Politics** → "government, party, election, minister, policy"

---

## Tech Stack

* **Python**
* **Pandas, NumPy** – Data handling
* **NLTK** – Preprocessing (tokenization, stopwords, lemmatization)
* **scikit-learn** – Vectorization (TF-IDF, CountVectorizer), NMF
* **Gensim** – LDA implementation
* **pyLDAvis** – Interactive LDA topic visualization
* **Matplotlib, Seaborn** – Data visualization

---

## Key Takeaways

* Topic modeling is an effective way to **discover latent themes** in text data without labels.
* **Preprocessing quality** significantly impacts the clarity of extracted topics.
* **TF-IDF + NMF** often yields sharper topics, while **LDA + pyLDAvis** provides intuitive interpretability.
* This workflow can be adapted to other text datasets for **insights, clustering, and knowledge discovery**.
