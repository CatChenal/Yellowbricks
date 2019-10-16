#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support as PRFS

from sklearn.svm import (LinearSVC,
                         NuSVC,
                         SVC)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import (LogisticRegressionCV,
                                  LogisticRegression,
                                  SGDClassifier)

from sklearn.ensemble import (BaggingClassifier,
                              ExtraTreesClassifier,
                              RandomForestClassifier)

from yellowbrick.classifier import ClassificationReport
from yellowbrick.datasets import load_mushroom
from sklearn.datasets import load_iris


def score_model(X, y, estimator, encode=True, **kwargs):
    """
    Test various estimators.
    From Yellowbrick example, amended.
    """
    y = LabelEncoder().fit_transform(y)
    
    if encode:
        model = Pipeline([('one_hot_encoder', OneHotEncoder()),
                          ('estimator', estimator)])
    else:
        model = Pipeline([('estimator', estimator)])
        
    # Instantiate the classification model and visualizer
    model.fit(X, y, **kwargs)

    expected  = y
    predicted = model.predict(X)

    # return model name, (P, R, Fscore, Support):
    return estimator.__class__.__name__, PRFS(expected, predicted)


def visualize_model(X, y, estimator, **kwargs):
    """
    Test various estimators.
    """
    y = LabelEncoder().fit_transform(y)
    model = Pipeline([
        ('one_hot_encoder', OneHotEncoder()),
        ('estimator', estimator)
    ])

    # Instantiate the classification model and visualizer
    visualizer = ClassificationReport(
        model, classes=['edible', 'poisonous'],
        cmap="YlGn", size=(600, 360), **kwargs
    )
    visualizer.fit(X, y)
    visualizer.score(X, y)
    visualizer.poof()


def get_models():
    #LinearSV{C,R}: max_iter=1000, tol=0.0001
    # removed LinearSVC(max_iter=1200): due to issue
    # https://github.com/scikit-learn/scikit-learn/issues/11536

    return [SVC(gamma='auto'),
            NuSVC(gamma='auto'),
            SGDClassifier(max_iter=100, tol=1e-3),
            KNeighborsClassifier(),
            LogisticRegression(solver='lbfgs', max_iter=500, multi_class='auto'),
            LogisticRegressionCV(cv=3, max_iter=300, multi_class='auto'),
            BaggingClassifier(),
            ExtraTreesClassifier(n_estimators=300),
            RandomForestClassifier(n_estimators=300)]


def get_mushroom_data():
    X, y = load_mushroom()
    labels = y.unique().tolist()
    return X, y, labels


def get_iris_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    labels = iris.target_names
    return X, y, labels


def yellowbrick_model_evaluation_report(X, y, models):
    for model in models:
        visualize_model(X, y, model)


# Alternates ...............................................
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# For df highlighting functions:
MAX_BGC = 'palegreen'
MIN_BGC = 'lightpink'


def get_scores_dict(models, X, y, labels, encode=True):
    """
    labels = class labels
    """
    results = {}
    for model in models:
        mdl, scores = score_model(X, y, model, encode=encode)
        # output per class:
        data_cls = {}
        for i, c in enumerate(labels):
            data_cls[c] = {'P':scores[0][i],
                           'R':scores[1][i],
                           'F1':scores[2][i],
                           'Support':scores[3][i]}
        results[mdl] = data_cls
    
    return results


def get_scores_df(models, X, y, labels, encode=True, to_style=False):
    d = get_scores_dict(models, X, y, labels, encode=encode)
    
    idx = 'columns' if to_style else 'index' 
    for k, v in d.items():      
        d[k] = pd.DataFrame.from_dict(v, idx)

    return pd.concat(d, axis=0)


def highlight_max(s):
    is_max = s == s.max()
    txt = 'background-color: {}'.format(MAX_BGC)
    return [txt if v else '' for v in is_max]


def highlight_min(s):
    is_min = s == s.min()
    txt = 'background-color: {}'.format(MIN_BGC)
    return [txt if v else '' for v in is_min]


def with_style(df, caption):
    styles = [dict(selector='th',
                   props=[('background-color', '#f7f7f9'),
                          ('text-align', 'right')]),
              dict(selector='caption',
                   props=[('caption-side', 'right'),
                          ('font-weight', 'bold'),
                          ('font-size', 'large')])
              ]
    df = df.style.set_table_styles(styles)\
                 .apply(highlight_min)\
                 .apply(highlight_max)\
                 .format('{:.6f}')\
                 .set_caption(caption)
    return df


def model_evaluation_report_tbl(models, X, y, labels, caption, encode=True):
    """
    Output a styled df as in model_evaluation_report_tbl_from_df(), starting 
    with models instances and data.
    """
    # get the scores:
    df = get_scores_df(models, X, y, labels, encode=encode, to_style=True)
    
    # Flip level1 to columns:
    df = df.unstack()

    # save Support values:
    sups = df.loc[:, (df.columns.get_level_values(0),
                      df.columns.get_level_values(1) == 'Support')].values[0]

    # Drop Support columns:
    df.drop(labels='Support', axis=1, level=1, inplace=True)

    # Create new col names -> levels[0]
    new_lev_0 = []
    for i, c in enumerate(df.columns.levels[0]):
        new_lev_0.append('{} (Support: {:.0f})'.format(c.title(), sups[i]))

    # Reset col index. Note: dropping the Support col did not
    # change the index, so [:-1] excludes it.
    mdx = pd.MultiIndex.from_product([new_lev_0,
                                      df.columns.levels[1][:-1]])
    df.columns = mdx

    # Style df:
    return with_style(df, caption)


def model_evaluation_report_bar(models, X, y, labels, xlim_to_1=False, encode=True):
    """
    
    """
    dfm = get_scores_df(models, X, y, labels, encode=encode)
    
    n_cats = len(dfm.index.levels[1])

    fig, axes = plt.subplots(nrows=1, ncols=n_cats,
                             sharey=True,
                             figsize=(4.5*n_cats, 6))

    # iter over classes
    for i, cat in enumerate(dfm.index.levels[1]):
        df = dfm[dfm.index.get_level_values(1) == cat].reset_index(level=1, drop=True)

        max_i = df.max(axis=0)[:-1]
        max_i_mdl = [df[df[df.columns[i]] == v].index.tolist() for i, v in enumerate(max_i)]

        # plot
        bars_i = df[df.columns[:-1]].plot(kind='barh', ax=axes[i])

        r, c = df.shape
        n_bars = r * (c-1)
        bar_cols = [bars_i.patches[i*r].get_facecolor() for i in range(c-1)]
        axes[i].vlines(max_i, -0.5, r,
                       color=bar_cols,
                       linestyle='dashed', lw=1.5)

        sup_i = int(df.loc[df.index[0],'Support'])
        axes[i].set_title('{} (support: {})'.format(cat.title(), sup_i),
                          size='large')
        if xlim_to_1:
            axes[i].set_xlim((0, 1))
        axes[i].grid(which='major', axis='y')
        axes[i].legend([])

    axes[0].tick_params(axis='y', labelsize=14)

    plt.legend(bbox_to_anchor=(0.5, .96),
               bbox_transform=plt.gcf().transFigure,
               loc='center',
               ncol=3,
               fontsize='medium') 

    plt.show();


"""
    Next two functions:
    Adapted/simplified implementation of these two functions from:
    https://github.com/deepmind/bsuite/bsuite/experiments/summary_analysis.py
"""
def _radar(df,
           ax,
           label,
           all_tags,
           alpha=0.15,
           edge_alpha=0.65,
           zorder=2,
           edge_style='-'):
    """
    Plot utility for generating the underlying radar plot.
    """
    values = df[label].values
    values = np.maximum(values, 0.05)  # don't let radar collapse to 0.
    values = np.concatenate((values, [values[0]]))

    angles = np.linspace(0, 2*np.pi, len(all_tags), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    ax.plot(angles, values, '-', 
            linewidth=1.8, 
            label=label,
            alpha=edge_alpha,
            zorder=zorder,
            linestyle=edge_style)
    
    ax.set_thetagrids(angles * 180/np.pi,
                      all_tags,
                      fontsize='medium')

    # To avoid text on top of gridlines, we flip horizontalalignment
    # based on label location
    text_angles = np.rad2deg(angles)
    for label, angle in zip(ax.get_xticklabels()[:-1], text_angles[:-1]):
        if 90 <= angle <= 270:
            label.set_horizontalalignment('right')
        else:
            label.set_horizontalalignment('left')
            
    return ax


def scores_radar_plot(df_scores):
    """ 
    df_scores: output of get_scores_df(models, X, y, labels)
    """
    fig = plt.figure(figsize=(4.2, 4.2), facecolor='white')

    ax = fig.add_subplot(1,1,1, polar=True)
    try:
        ax.set_axis_bgcolor('white')
    except AttributeError:
        ax.set_facecolor('white')

    all_tags = df_scores.index.tolist()
    
    for i, c in enumerate(df_scores.columns.tolist()):
        _radar(df_scores, ax, c, all_tags)
        
    legend = ax.legend(loc=(1.15, 0.), ncol=1,
                       title='Scores')
    
    plt.setp(legend.get_title(),
             fontname='serif',
             fontsize='medium',
             color='k',
             alpha=0.95)
    plt.setp(legend.texts,
             fontname='serif',
             fontsize='medium',
             color='k',
             alpha=0.8)
    plt.setp(ax.xaxis.get_gridlines(),
             color='grey',
             alpha=0.95,
             linestyle=':',
             lw=1)
    plt.setp(ax.yaxis.get_gridlines(),
             color='grey',
             alpha=0.95,
             linestyle=':',
             lw=1)

    plt.xticks(color='#11557c', fontsize='medium')
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.25, 0.5, 0.75, 1],
               ['', '.25', '.5', '.75', '1'],
               color='k', alpha=0.9,
               fontsize='medium')

    if df_scores.index.name:
        ax.set_title(df_scores.index.name,
                     fontname='serif',
                     fontsize='medium',
                     fontweight='bold')
    return ax


def scores_radar_plot_example(dfm, cat='setosa'):
    supp = dfm.loc[(dfm.index.get_level_values(0)[0],
                    dfm.index.get_level_values(1) == cat),
                        'Support'].values[0]
    
    df = dfm.loc[dfm.index.get_level_values(1) == cat,
                 dfm.columns[:-1]].reset_index(level=1, drop=True)
    df.index.name = '{} (support: {})'.format(cat.title(), supp)
    
    ax = scores_radar_plot(df)
    
    '''
    # a bit too crowded with additional text:
    fig = plt.gcf()
    s = 'Radar plot of model selection scores for class {}'.format(cat.title())
    fig.text(0.5, -0.05, s,
             ha='center',
             fontsize='large')
    '''
    plt.show()
    
    
def generic_polar():
    xs = np.arange(10)
    ys = np.random.rand(10,3)

    rgrid=[0.25, 0.5, 0.75, 1.]

    mrk = ['bo', 'go', 'ro']
    fig = plt.figure(figsize=(5, 10))
    ax = plt.subplot(1, 1, 1, projection='polar')

    for c in np.arange(ys.shape[1]):
        for x, y in zip(xs, ys[:,c]):
            plt.polar(x, y, mrk[c])

    ax.grid(lw=0.5, color='0.9')
    ax.set_yticks(rgrid)
    plt.show()
