import streamlit as st
import pandas as pd
import numpy as np
#import os
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import json
import requests
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import subprocess
import sys
from pip._internal import main as pipmain

#pipmain(['install', "torch"])
#pipmain(['install', "torchmetrics"])
#pipmain(['install', "pytorch_lightning"])
#pipmain(['install', "pylab"])
#pipmain(['install', "transformers"])

# from model import *
#######3#model.py##########
# Pip Installing Dependencies
# !pip install torch -q
# !pip install watermark -q
# !pip install transformers -q
# !pip install --upgrade pytorch-lightning -q
# !pip install colored -q
# !pip install -U -q PyDrive -q

# Import Packages
# import pandas as pd
# import numpy as np
# from tqdm.auto import tqdm
import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, auroc
from torchmetrics.functional import f1_score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from nltk import ngrams
import re
import string

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, multilabel_confusion_matrix
# import seaborn as sns
#from pylab import rcParams
# from matplotlib import rc

# %matplotlib inline
# %config InlineBackend.figure_format='retina'
# RANDOM_SEED = 42
# sns.set(style='whitegrid', palette='muted', font_scale=1.2)
# HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
# sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
# rcParams['figure.figsize'] = 12, 8
# pl.seed_everything(RANDOM_SEED)

class TweetTagger(pl.LightningModule):
  def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased', return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.BCELoss()
  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output
  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}
  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss
  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss
  def training_epoch_end(self, outputs):
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)
    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)
    for i, name in enumerate(6):
      class_roc_auc = auroc(predictions[:, i], labels[:, i])
      self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )
    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )

# def get_model_predictions(tweet):
#     model = TweetTagger(n_classes=6, n_warmup_steps=140, n_training_steps=703)
#     loaded_model = TweetTagger(n_classes=6,
#                            n_warmup_steps=140,
#                            n_training_steps=703)
#
#     #cwd = os.getcwd() # getting current working directory
#     #print('This is the Current Directory: ')
#     #print(cwd + '/pytorch_model.pth')
#     loaded_model.load_state_dict(torch.load('pytorch_model.pth'))
#     loaded_model.eval()
#
#     BERT_MODEL_NAME = 'bert-base-cased'
#     tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
#
#     encoding = tokenizer.encode_plus(tweet, add_special_tokens=True, max_length=512,
#     return_token_type_ids=False, padding="max_length", return_attention_mask=True,
#     return_tensors='pt',)
#
#     _, test_prediction = loaded_model(encoding["input_ids"], encoding["attention_mask"])
#     test_prediction = test_prediction.flatten().detach()
#     prediction_values = [pred.item() for pred in test_prediction]
#     LABEL_COLUMNS = ['Neutral', 'General Criticsm', 'Disability Shaming', 'Racial Prejudice',
#                  'Sexism','LGBTQ+ Phobia']
#
#     result = []
#     for label, prediction in zip(LABEL_COLUMNS, prediction_values):
#         result.append([label, prediction])
#
#     return result

def generate_N_grams(text,ngram=1):
    words=[word for word in text.split(" ")]
    temp=zip(*[words[i:] for i in range(0,ngram)])
    ans=[' '.join(ngram) for ngram in temp]
    return ans

def count_category(tweet, loaded_model):
    #model = TweetTagger(n_classes=6, n_warmup_steps=140, n_training_steps=703)
    #loaded_model = TweetTagger(n_classes=6, n_warmup_steps=140, n_training_steps=703)
    # define sentiment lists for each color
    neutral_words = []
    general_criticism_words = []
    disability_shaming_words = []
    racist_words = []
    sexist_words = []
    lgbtq_words = []

    # remove punctutation
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # dictionary storing the count of each sentiment detected from BERT (increment by )0.5 because we use bigrams
    count_dict = {'neutral_count': 0, 'general_criticism_count': 0, 'disability_count': 0, 'racist_count': 0, 'sexist_count': 0, 'lgbtq_count': 0}
    # first and last unigrams
    unigrams = generate_N_grams(tweet, 1)

    BERT_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


    for unigram in unigrams:
        encoding = tokenizer.encode_plus(
        unigram,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
        )
        _, test_prediction = loaded_model(encoding["input_ids"], encoding["attention_mask"])
        test_prediction = test_prediction.flatten().detach().numpy()
    # print(f'unigram: {unigram}, test_prediction: {test_prediction}')
        if test_prediction[0] > 0.5:
            count_dict['neutral_count'] += 1
            neutral_words.append(unigram)
        elif test_prediction[1] > 0.5:
            count_dict['general_criticism_count'] += 1
            general_criticism_words.append(unigram)
        elif test_prediction [2] > 0.5:
            count_dict['disability_count'] += 1
            disability_shaming_words.append(unigram)
        elif test_prediction[3] > 0.5:
            count_dict['racist_count'] += 1
            racist_words.append(unigram)
        elif test_prediction[4] > 0.5:
            count_dict['sexist_count'] += 1
            sexist_words.append(unigram)
        elif test_prediction[5] > 0.5:
            count_dict['lgbtq_count'] += 1
            lgbtq_words.append(unigram)
    return count_dict


def return_distribution(test_comment):

    model = TweetTagger(n_classes=6, n_warmup_steps=140, n_training_steps=703)
    loaded_model = TweetTagger(n_classes=6, n_warmup_steps=140, n_training_steps=703)

    loaded_model.load_state_dict(torch.load('pytorch_model.pth'))
    loaded_model.eval()

    BERT_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    encoding = tokenizer.encode_plus(
    test_comment,
    add_special_tokens=True,
    max_length=512,
    return_token_type_ids=False,
    padding="max_length",
    return_attention_mask=True,
    return_tensors='pt',
  )
    _, test_prediction = loaded_model(encoding["input_ids"], encoding["attention_mask"])
    test_prediction = test_prediction.flatten().detach().numpy()

    # multiply the original outputs by the term frequency (TF) of each category

    count_dict = count_category(test_comment, loaded_model)


    neutral_score = count_dict['neutral_count']*test_prediction[0]
    general_criticism_score = count_dict['general_criticism_count']*test_prediction[1]
    weighted_disability_score = count_dict['disability_count']*test_prediction[2]
    weighted_racist_score = count_dict['racist_count']*test_prediction[3]
    weighted_sexist_score = count_dict['sexist_count']*test_prediction[4]
    weighted_lgbtq_score = count_dict['lgbtq_count']*test_prediction[5]

    #if neutral_score < 0.5 or general_criticism_score <0.5:
    LABEL_COLUMNS = ['neutral', 'general criticsm', 'disability shaming', 'racial prejudice',
                     'sexism','lgbtq+ phobia']

    weighted_test_prediction = np.array([neutral_score, general_criticism_score, weighted_disability_score,weighted_racist_score,weighted_sexist_score,weighted_lgbtq_score])

    result = []
    for label, prediction in zip(LABEL_COLUMNS, weighted_test_prediction):
      result.append([label, prediction])

    return result


###########################



header = st.container()
mission = st.container()
dataset = st.container()
models = st.container()
#ale changed this line
java = st.container()
resource = st.container()

with header:
    #Insert  Title
    st.title("Welcome to our Capstone Project!")
    image = Image.open('tone_log.png')
    st.image(image, caption = "Toning down the bad vibes")

with mission:
    st.title("Mission Statement:")
    st.text("Promoting empathy among Twitter Users to reduce offensive content that harms the wellness of users")

with dataset:
    sentence = st.text_input('Input your sentence here:')
    if sentence:
        answer = return_distribution(sentence)
        #st.write(answer)
    else:
        answer = [['Neutral', 1.0], ['General Criticism', 0],
        ['Disability Shaming', 0], ['Sexism', 0],
        ['Racial Prejudice', 0], ['LGBTQ+ Phobia', 0]
        ]
    st.text("""
    The data is composed of about 24,000 tweets derived from the Kaggle Hate
    Speech and Offensive Language Dataset.The original dataset was conceived
    to be used to research hate speech such as racial, homophobic, sexist,
    and general offensive language. It had the following columns that we
    later modify: hate_speech, offensive_language, and neither.
    Since we wanted to help users reflect deeper about the type of
    offensive language they may be putting out into the world, we decided
    to alter the dataset in the following ways:
	1.	We began by creating the following columns: 'Neutral',
    'General Criticism', 'Disability Shaming', 'Sexism','Racial Prejudice',
    and 'LGBTQ+ Phobic'.
	2.	Since these new labels were not present in the original dataset,
    we needed to relabel using our new columns.
	3.	Language is fundamentally complex and context is important to discern
    more subtle offensive sentences and phrases. We wanted to be mindful,
    accurate, and consistant with our relabeling process. To do this we created
    a labeling methodology [link here] that each one of our members followed
    while manually reading and relabeling thousands of tweets.
	4.	We then fed our newly relabeled into our PyTorch model where we train
    the machine learning algorithm to recognize hate speech and predict the
    type of offensive language.

    Here's a preview of our dataset using real tweets:""")
    data = pd.read_csv("multi_label_new.csv", encoding = "ISO-8859-1")
    answer.insert(0, ['Task', 'Hours per Day'])


    st.write(data.tail(10))
    # pred = model.get_model_predictions("I hate james a lot")
    # st.text(pred)

#Writes the html/css/javascript: Mostly for the donut chart
#ale changed this too
with java:
    components.html(
        """
        <section>
        <div class="donut-chart", style = "position:relative; background-color: transparent;">
      <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
          <script type="text/javascript">
            google.charts.load("current", {packages:["corechart"]});
            google.charts.setOnLoadCallback(drawChart);
            function drawChart() {
              var data = google.visualization.arrayToDataTable(""" + str(answer) + """);
              var options = {
                title: 'Tone Representation',
                pieHole: 0.4,
                colors: ['#36d8ff', '#529ffc', '#31356e', '#66757f', '#5F9EA0', '#96DED1']
              };
              var chart = new google.visualization.PieChart(document.getElementById('donutchart'));
              chart.draw(data, options);
            }
          </script>
          <div id="donutchart" style="width: 700px; height: 350px;"></div></p>
        </div>
        </section>
        """,
        height=600,
    )

# Resources page #
with resource:
    st.title("Resources")
    st.write("""
    Given the nature of our product, we were hoping that we could
    provide resources for users to learn more about ways to combat or
    educate themselves regarding offensive language. Below are some
    resources we provided as a stepping stone to learn more about LGBTQ+
    community, gender equity, Disability awareness, and racial equality.""")

    st.write("Sexism:")
    st.write("Britannica - Sexism Definition:[link](https://www.britannica.com/topic/sexism)")
    st.write("European Institute for Gender Equality - What is Sexism: [link] (https://eige.europa.eu/publications/sexism-at-work-handbook/part-1-understand/what-sexism)")
    st.write("Human Rights Channel - Sexism: See it. Name it. Stop it: [link] (https://human-rights-channel.coe.int/stop-sexism-en.html)")
    st.write("Science Direct - Sexism: [link] (https://www.sciencedirect.com/topics/psychology/sexism)")

    st.write("Racial Prejudice:")
    st.write("United Nations Declaration on Race and Racial Prejudice: [link] (https://www.ohchr.org/en/instruments-mechanisms/instruments/declaration-race-and-racial-prejudice")
    st.write("U.S. Equal Employment Opportunity Commission Race/Color Discrimination: [link] (https://www.eeoc.gov/racecolor-discrimination)")
    st.write("Alberta Civil Liberties Research Centre - Racism: [link] (https://www.aclrc.com/racism)")
    st.write("The National Association of School Psychologists (NASP) - Prejudice, Discrimination, and Racism: [link] (https://www.nasponline.org/x26830.xml)")
    st.write("University of Minnesota - Prejudice – Sociology - Publishing Services:[link] (https://open.lib.umn.edu/sociology/chapter/10-3-prejudice/)")

    st.write("Disability")
    st.write("The Lakeshore West Michigan’s How to respect people with disabilities: [link] (https://www.secondwavemedia.com/lakeshore/features/Persons_First_Language_respects_people_with_disabilities.aspx)")
    st.write("Etiquette: Interacting with People with Disabilities: [link] (https://www.respectability.org/inclusion-toolkits/etiquette-interacting-with-people-with-disabilities/)")
    st.write("Illinois Department of Human Services - A Guide to Interacting with People with Disabilities: [link] (https://www.dhs.state.il.us/page.aspx?item=32276)")
    st.write("New York State Department of Health: Disability Etiquette Treat: Everyone with Respect: [link] (https://www.health.ny.gov/publications/0951.pdf)")
    st.write("Capital Women’s Care’s Showing Acceptance and Respect for Those with Disabilities: [link] (https://www.cwcare.net/news/showing-acceptance-and-respect-those-disabilities)")
