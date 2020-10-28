import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import requests, zipfile, io
from pla import chat
import glob, os

interesting_corpora = [
    "Braunwald",
    "NewmanRatner",
    "Sachs",
    "Snow",
    "Soderstrom",
    "Providence",
    "Paris",
    "Lyon",
    "Stuggart",
    "TAKI",
    "Lacerda",
    "Garmann",
]

vc = pd.read_csv("vc.csv")
links = {}
for index, row in vc.iterrows():
    links[row["Corpus"]] = row["link"]


def open(corpus):
    """
    Opens the CHILDES webpage for a [corpus], entered as a string of the name of the corpus
    (ie 'Bates', 'Brown', etc)
    """
    os.system('open ""' + links[corpus])


def download_zip(corpus):
    """
    Downloads and unzips the [corpus] to the current directory.
    """
    link = vc[corpus]
    link = link.replace("html", "zip")
    link = link.replace("access", "data")
    r = requests.get(link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()


def label_start(row):
    """
    Helper function for pandize
    """
    if row["Duration"] == None:
        return None
    else:
        return row["Duration"][0] / 1000


def label_end(row):
    """
    Helper function for pandize
    """
    if row["Duration"] == None:
        return None
    else:
        return row["Duration"][1] / 1000


def phonemize(row, tiers):
    """
    Helper function for pandize
    """
    if "%pho" in tiers[row["Index"]]:
        return tiers[row["Index"]]["%pho"]
    return None


def pandize(filepath):
    """
    Takes a .cha file of a given [filepath] and creates a pandas data frame, where a row
    represents a vocalizaiton, with columns Speaker, Vocalization, Start, and End (representing
    the beginning and ending time stamps of the vocalization).
    """
    reader = chat.SingleReader(filepath)
    tiers = reader.index_to_tiers()
    s = list(reader.participants().keys())
    s = ", ".join(s)
    u = reader.utterances(time_marker=True)
    df = pd.DataFrame(u, columns=["Speaker", "Vocalization", "Duration"])
    df["Participants"] = s
    df["Corpus"] = filepath[: filepath.find("/")]
    df["Index"] = df.index
    df["file"] = filepath
    df["onset"] = df.apply(lambda row: label_start(row), axis=1)
    df["offset"] = df.apply(lambda row: label_end(row), axis=1)
    df["Pho"] = df.apply(lambda row: phonemize(row, tiers), axis=1)
    df = df.drop(["Duration", "Index"], axis=1)
    return df


def combine(arr):
    """
    Returns a dataframe representing pandize of every corpus in arr combined into
    one dataframe.
    """
    d1 = pandize(arr[0])
    errs = []
    if len(arr) < 1:
        return d1
    for i in range(1, len(arr)):
        try:
            d1 = d1.append(pandize(arr[i]))
        except:
            errs.append(arr[i])
    print(errs)
    return d1


def full_corpus(corpus):
    """
    Similar to combine, but returns a dataframe representing pandize for every
    file in a corpus all combined into one dataframe.
    """
    f = []
    for root, dirs, files in os.walk(corpus):
        for file in files:
            if file.endswith(".cha"):
                f.append(str((os.path.join(root, file))))
    return combine(f)


def mult_corpora(corpora):
    """
    Returns a dataframe that is full_corpus of every corpus in corpora
    """
    d1 = full_corpus(corpora[0])
    for i in range(1, len(corpora)):
        d1 = d1.append(full_corpus(corpora[i]))
    return d1


def turns(df):
    """
    Returns arrays representing onset and offset times for mothers and children
    given a dataframe df, and returns the final onset time
    """
    chi = []
    mot = []
    m = 0
    for index, row in df.iterrows():
        if row["Speaker"] == "CHI":
            try:
                chi.append((row["onset"], row["offset"]))
                m = max(m, row["onset"])
            except:
                pass
        elif row["Speaker"] == "MOT":
            try:
                mot.append((row["onset"], row["offset"]))
                m = max(m, row["onset"])
            except:
                pass
    return (chi, mot, m)


def dual_stream_viz(df, title):
    """
    Plots turn-taking for dataframe df
    """
    alpha = 1
    maxm = turns(df)[2]
    par = df.loc[0, "Participants"].split(", ")
    variables = []
    for i in range(len(par)):
        variables.append(df.loc[df["Speaker"] == par[i]])
    colors = ["red", "blue", "green", "brown"]
    for j in range(min(len(variables), 4)):
        print(j)
        variables[j]["Color"] = colors[j]
    variable_1 = variables[0]
    try:
        variable_2 = variables[1]
    except:
        print("no 2")
    try:
        variable_3 = variables[2]
    except:
        print("no 3")
    try:
        variable_4 = variables[3]
    except:
        print("no 4")
    begin = 0
    qtr = (maxm - begin) / 4  #  divide into quarters
    columns = ["onset", "offset"]
    qtr1 = pd.DataFrame(
        np.array(
            [
                [begin, qtr + begin],
            ]
        ),
        columns=["onset", "offset"],
    )  #  1st quarter to visualize
    qtr2 = pd.DataFrame(
        np.array(
            [
                [qtr + 1 + begin, (qtr * 2) + begin],
            ]
        ),
        columns=columns,
    )  #  2nd quarter to visualize
    qtr3 = pd.DataFrame(
        np.array(
            [
                [(qtr * 2 + 1) + begin, (qtr * 3) + begin],
            ]
        ),
        columns=columns,
    )  #  3rd quarter to visualize
    qtr4 = pd.DataFrame(
        np.array(
            [
                [(qtr * 3 + 1) + begin, (qtr * 4) + begin],
            ]
        ),
        columns=columns,
    )  #  4th quarter to visualize
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(18.5, 12))
    for i in variables:
        ax1.broken_barh(
            list(
                zip(
                    i["onset"].values,
                    (i["offset"] - i["onset"]).values,
                )
            ),
            (2.5, 0.99),
            color=i["Color"],
            edgecolor="black",
            alpha=alpha,
        )
        ax1.set_xlim(qtr1.at[0, "onset"], qtr1.at[0, "offset"])
        ax2.broken_barh(
            list(
                zip(
                    i["onset"].values,
                    (i["offset"] - i["onset"]).values,
                )
            ),
            (2.5, 0.99),
            color=i["Color"],
            edgecolor="black",
            alpha=alpha,
        )
        ax2.set_xlim(qtr2.at[0, "onset"], qtr2.at[0, "offset"])
        ax3.broken_barh(
            list(
                zip(
                    i["onset"].values,
                    (i["offset"] - i["onset"]).values,
                )
            ),
            (2.5, 0.99),
            color=i["Color"],
            edgecolor="black",
            alpha=alpha,
        )
        ax3.set_xlim(qtr3.at[0, "onset"], qtr3.at[0, "offset"])
        ax4.broken_barh(
            list(
                zip(
                    i["onset"].values,
                    (i["offset"] - i["onset"]).values,
                )
            ),
            (2.5, 0.99),
            color=i["Color"],
            edgecolor="black",
            alpha=alpha,
        )
        ax4.set_xlim(qtr4.at[0, "onset"], qtr4.at[0, "offset"])
    plt.savefig("{}.png".format(title))


def plot_corpus(corpus):
    """
    Strives to make a plot for every file in corpus.
    """
    f = []
    for root, dirs, files in os.walk(corpus):
        for file in files:
            if file.endswith(".cha"):
                f.append(str((os.path.join(root, file))))
    pandizes = []
    viz = []
    for i in f:
        title = i.replace("/", ".")
        try:
            x = pandize(i)
        except:
            pandizes += [i]
        try:
            dual_stream_viz(x, title)
        except:
            viz += [i]
    v = []
    for i in viz:
        if i not in pandizes:
            v.append(i)
    print(pandizes, v)
