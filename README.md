# Visualizing the Evaluation of ML Models with Yellowbrick and alternate implementations  
>[Yellowbrick](http://www.scikit-yb.org/en/latest/index.html) extends the Scikit-Learn API to make model selection and hyperparameter tuning easier.
---

**Note**: one alternate visualization uses PyViz.Panel. If not installed, un-comment and run this command:
#conda install -c pyviz panel

---
## The code producing the following visuals is in the module `model_evaluation_reports.py`
```python
import model_evaluation_reports as rpts
```

---
# Yellowbrick binary classification example:
http://www.scikit-yb.org/en/latest/tutorial.html

---
## Model evaluation report using Yellowbrick `ClassificationReport()`:
Yellowbrick's visual report returns a matrix of P, R, and F1 scores for **each** model.  
It is indeed very neat, but in my opinion, not very practical since the goal of the visualization is to enable picking "the best" model...

### With lots of models &rarr; lots of scolling!
```python
# Mushroom dataset & models list:
X, y, labels = rpts.get_mushroom_data()
models = rpts.get_models()

rpts.run_yellowbrick_model_evaluation_report(X, y, models)
```

![png](images/yellowbrick_cls_model_selection_9_0.png)

![png](images/yellowbrick_cls_model_selection_9_1.png)

![png](images/yellowbrick_cls_model_selection_9_2.png)

![png](images/yellowbrick_cls_model_selection_9_3.png)

![png](images/yellowbrick_cls_model_selection_9_4.png)

![png](images/yellowbrick_cls_model_selection_9_5.png)

![png](images/yellowbrick_cls_model_selection_9_6.png)

![png](images/yellowbrick_cls_model_selection_9_7.png)

![png](images/yellowbrick_cls_model_selection_9_8.png)


---
# Alternate visualizations

**Note**: The table view uses [PyViz.Panel](https://panel.pyviz.org/)

---
## Gather the evaluation scores into a DataFrame:
```python
# Mushroom dataset & models list:
X, y, labels = rpts.get_mushroom_data()
models = rpts.get_models()

dfm_mushrooms = rpts.get_scores_df(models, X, y, labels)
```

---
## Table output 
### &rarr; using `run_alternate_model_evaluation_report_tbl()`, retaining Support values:
```python
rpts.run_alternate_model_evaluation_report_tbl(dfm_mushrooms) # green: max; pink: min
```
<div id='1001'>
  <div class="bk-root" id="6fc9a8d6-3c6d-4ecf-b224-674bbf5941d0" data-root-id="1001"></div>
</div>
<script type="application/javascript">
    function msg_handler(msg) {
      var metadata = msg.metadata;
      var buffers = msg.buffers;
      var msg = msg.content.data;
      if ((metadata.msg_type == "Ready")) {
        if (metadata.content) {
          console.log("Python callback returned following output:", metadata.content);
        }
      } else if (metadata.msg_type == "Error") {
        console.log("Python failed with the following traceback:", metadata.traceback)
      } else {
        
var plot_id = "1001";
if (plot_id in window.PyViz.plot_index) {
  var plot = window.PyViz.plot_index[plot_id];
} else {
  var plot = Bokeh.index[plot_id];
}

if (plot_id in window.PyViz.receivers) {
  var receiver = window.PyViz.receivers[plot_id];
} else if (Bokeh.protocol === undefined) {
  return;
} else {
  var receiver = new Bokeh.protocol.Receiver();
  window.PyViz.receivers[plot_id] = receiver;
}

if ((buffers != undefined) && (buffers.length > 0)) {
  receiver.consume(buffers[0].buffer)
} else {
  receiver.consume(msg)
}

const comm_msg = receiver.message;
if ((comm_msg != null) && (Object.keys(comm_msg.content).length > 0)) {
  plot.model.document.apply_json_patch(comm_msg.content, comm_msg.buffers)
}

      }
    }
    if ((window.PyViz == undefined) || (!window.PyViz.comm_manager)) {
      console.log("Could not find comm manager")
    } else {
      window.PyViz.comm_manager.register_target('1001', '03683f0f02e942639adf40271d26d559', msg_handler);
    }
    
(function(root) {
  function embed_document(root) {
    
  var docs_json = {"4ef0924c-19fd-4422-8f9f-8baf420825de":{"roots":{"references":[{"attributes":{"align":"end","children":[{"id":"1004","type":"Div"}],"margin":[0,0,0,0],"max_height":45,"name":"Row00009"},"id":"1003","type":"Row"},{"attributes":{"margin":[5,5,5,5],"name":"HTML00017","text":"&lt;style  type=&quot;text/css&quot; &gt;\n    #T_8afac188_ef64_11e9_b085_00155d01c508 th {\n          background-color: #f7f7f9;\n          text-align: right;\n    }    #T_8afac188_ef64_11e9_b085_00155d01c508 caption {\n          caption-side: right;\n          font-weight: bold;\n    }    #T_8afac188_ef64_11e9_b085_00155d01c508row0_col1 {\n            : ;\n            background-color:  palegreen;\n        }    #T_8afac188_ef64_11e9_b085_00155d01c508row0_col2 {\n            : ;\n            background-color:  palegreen;\n        }    #T_8afac188_ef64_11e9_b085_00155d01c508row1_col0 {\n            : ;\n            background-color:  palegreen;\n        }    #T_8afac188_ef64_11e9_b085_00155d01c508row2_col1 {\n            background-color:  lightpink;\n            : ;\n        }    #T_8afac188_ef64_11e9_b085_00155d01c508row3_col0 {\n            background-color:  lightpink;\n            : ;\n        }    #T_8afac188_ef64_11e9_b085_00155d01c508row6_col0 {\n            : ;\n            background-color:  palegreen;\n        }    #T_8afac188_ef64_11e9_b085_00155d01c508row7_col2 {\n            background-color:  lightpink;\n            : ;\n        }&lt;/style&gt;&lt;table id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508&quot; &gt;&lt;thead&gt;    &lt;tr&gt;        &lt;th class=&quot;col_heading level0 col0&quot; &gt;P&lt;/th&gt;        &lt;th class=&quot;col_heading level0 col1&quot; &gt;R&lt;/th&gt;        &lt;th class=&quot;col_heading level0 col2&quot; &gt;F1&lt;/th&gt;    &lt;/tr&gt;&lt;/thead&gt;&lt;tbody&gt;\n                &lt;tr&gt;\n                                &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row0_col0&quot; class=&quot;data row0 col0&quot; &gt;0.711264&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row0_col1&quot; class=&quot;data row0 col1&quot; &gt;0.687101&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row0_col2&quot; class=&quot;data row0 col2&quot; &gt;0.698974&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                                &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row1_col0&quot; class=&quot;data row1 col0&quot; &gt;0.730305&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row1_col1&quot; class=&quot;data row1 col1&quot; &gt;0.648787&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row1_col2&quot; class=&quot;data row1 col2&quot; &gt;0.687136&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                                &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row2_col0&quot; class=&quot;data row2 col0&quot; &gt;0.697768&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row2_col1&quot; class=&quot;data row2 col1&quot; &gt;0.622733&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row2_col2&quot; class=&quot;data row2 col2&quot; &gt;0.658119&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                                &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row3_col0&quot; class=&quot;data row3 col0&quot; &gt;0.647233&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row3_col1&quot; class=&quot;data row3 col1&quot; &gt;0.669221&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row3_col2&quot; class=&quot;data row3 col2&quot; &gt;0.658043&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                                &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row4_col0&quot; class=&quot;data row4 col0&quot; &gt;0.647407&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row4_col1&quot; class=&quot;data row4 col1&quot; &gt;0.669732&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row4_col2&quot; class=&quot;data row4 col2&quot; &gt;0.658380&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                                &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row5_col0&quot; class=&quot;data row5 col0&quot; &gt;0.700914&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row5_col1&quot; class=&quot;data row5 col1&quot; &gt;0.646488&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row5_col2&quot; class=&quot;data row5 col2&quot; &gt;0.672602&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                                &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row6_col0&quot; class=&quot;data row6 col0&quot; &gt;0.730305&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row6_col1&quot; class=&quot;data row6 col1&quot; &gt;0.648787&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row6_col2&quot; class=&quot;data row6 col2&quot; &gt;0.687136&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                                &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row7_col0&quot; class=&quot;data row7 col0&quot; &gt;0.665099&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row7_col1&quot; class=&quot;data row7 col1&quot; &gt;0.650319&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row7_col2&quot; class=&quot;data row7 col2&quot; &gt;0.657626&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                                &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row8_col0&quot; class=&quot;data row8 col0&quot; &gt;0.673082&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row8_col1&quot; class=&quot;data row8 col1&quot; &gt;0.652107&lt;/td&gt;\n                        &lt;td id=&quot;T_8afac188_ef64_11e9_b085_00155d01c508row8_col2&quot; class=&quot;data row8 col2&quot; &gt;0.662429&lt;/td&gt;\n            &lt;/tr&gt;\n    &lt;/tbody&gt;&lt;/table&gt;"},"id":"1011","type":"HTML"},{"attributes":{"children":[{"id":"1003","type":"Row"},{"id":"1005","type":"Row"}],"margin":[0,0,0,0],"name":"Column00013"},"id":"1002","type":"Column"},{"attributes":{"margin":[5,5,5,5],"name":"HTML00010","text":"&lt;style  type=&quot;text/css&quot; &gt;\n    #T_8af3e398_ef64_11e9_a5fd_00155d01c508 th {\n          background-color: #f7f7f9;\n          text-align: right;\n    }    #T_8af3e398_ef64_11e9_a5fd_00155d01c508 caption {\n          caption-side: right;\n          font-weight: bold;\n    }    #T_8af3e398_ef64_11e9_a5fd_00155d01c508row0_col0 {\n            : ;\n            background-color:  palegreen;\n        }    #T_8af3e398_ef64_11e9_a5fd_00155d01c508row1_col1 {\n            : ;\n            background-color:  palegreen;\n        }    #T_8af3e398_ef64_11e9_a5fd_00155d01c508row1_col2 {\n            : ;\n            background-color:  palegreen;\n        }    #T_8af3e398_ef64_11e9_a5fd_00155d01c508row2_col0 {\n            background-color:  lightpink;\n            : ;\n        }    #T_8af3e398_ef64_11e9_a5fd_00155d01c508row3_col1 {\n            background-color:  lightpink;\n            : ;\n        }    #T_8af3e398_ef64_11e9_a5fd_00155d01c508row3_col2 {\n            background-color:  lightpink;\n            : ;\n        }    #T_8af3e398_ef64_11e9_a5fd_00155d01c508row4_col1 {\n            background-color:  lightpink;\n            : ;\n        }    #T_8af3e398_ef64_11e9_a5fd_00155d01c508row6_col1 {\n            : ;\n            background-color:  palegreen;\n        }    #T_8af3e398_ef64_11e9_a5fd_00155d01c508row6_col2 {\n            : ;\n            background-color:  palegreen;\n        }&lt;/style&gt;&lt;table id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508&quot; &gt;&lt;thead&gt;    &lt;tr&gt;        &lt;th class=&quot;blank level0&quot; &gt;&lt;/th&gt;        &lt;th class=&quot;col_heading level0 col0&quot; &gt;P&lt;/th&gt;        &lt;th class=&quot;col_heading level0 col1&quot; &gt;R&lt;/th&gt;        &lt;th class=&quot;col_heading level0 col2&quot; &gt;F1&lt;/th&gt;    &lt;/tr&gt;&lt;/thead&gt;&lt;tbody&gt;\n                &lt;tr&gt;\n                        &lt;th id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508level0_row0&quot; class=&quot;row_heading level0 row0&quot; &gt;BaggingClassifier&lt;/th&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row0_col0&quot; class=&quot;data row0 col0&quot; &gt;0.717807&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row0_col1&quot; class=&quot;data row0 col1&quot; &gt;0.740494&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row0_col2&quot; class=&quot;data row0 col2&quot; &gt;0.728974&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                        &lt;th id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508level0_row1&quot; class=&quot;row_heading level0 row1&quot; &gt;ExtraTreesClassifier&lt;/th&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row1_col0&quot; class=&quot;data row1 col0&quot; &gt;0.703983&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row1_col1&quot; class=&quot;data row1 col1&quot; &gt;0.777091&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row1_col2&quot; class=&quot;data row1 col2&quot; &gt;0.738733&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                        &lt;th id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508level0_row2&quot; class=&quot;row_heading level0 row2&quot; &gt;KNeighborsClassifier&lt;/th&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row2_col0&quot; class=&quot;data row2 col0&quot; &gt;0.680925&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row2_col1&quot; class=&quot;data row2 col1&quot; &gt;0.749049&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row2_col2&quot; class=&quot;data row2 col2&quot; &gt;0.713364&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                        &lt;th id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508level0_row3&quot; class=&quot;row_heading level0 row3&quot; &gt;LogisticRegression&lt;/th&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row3_col0&quot; class=&quot;data row3 col0&quot; &gt;0.682209&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row3_col1&quot; class=&quot;data row3 col1&quot; &gt;0.660646&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row3_col2&quot; class=&quot;data row3 col2&quot; &gt;0.671254&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                        &lt;th id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508level0_row4&quot; class=&quot;row_heading level0 row4&quot; &gt;LogisticRegressionCV&lt;/th&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row4_col0&quot; class=&quot;data row4 col0&quot; &gt;0.682544&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row4_col1&quot; class=&quot;data row4 col1&quot; &gt;0.660646&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row4_col2&quot; class=&quot;data row4 col2&quot; &gt;0.671416&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                        &lt;th id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508level0_row5&quot; class=&quot;row_heading level0 row5&quot; &gt;NuSVC&lt;/th&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row5_col0&quot; class=&quot;data row5 col0&quot; &gt;0.693262&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row5_col1&quot; class=&quot;data row5 col1&quot; &gt;0.743346&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row5_col2&quot; class=&quot;data row5 col2&quot; &gt;0.717431&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                        &lt;th id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508level0_row6&quot; class=&quot;row_heading level0 row6&quot; &gt;RandomForestClassifier&lt;/th&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row6_col0&quot; class=&quot;data row6 col0&quot; &gt;0.703983&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row6_col1&quot; class=&quot;data row6 col1&quot; &gt;0.777091&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row6_col2&quot; class=&quot;data row6 col2&quot; &gt;0.738733&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                        &lt;th id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508level0_row7&quot; class=&quot;row_heading level0 row7&quot; &gt;SGDClassifier&lt;/th&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row7_col0&quot; class=&quot;data row7 col0&quot; &gt;0.681257&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row7_col1&quot; class=&quot;data row7 col1&quot; &gt;0.695342&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row7_col2&quot; class=&quot;data row7 col2&quot; &gt;0.688228&lt;/td&gt;\n            &lt;/tr&gt;\n            &lt;tr&gt;\n                        &lt;th id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508level0_row8&quot; class=&quot;row_heading level0 row8&quot; &gt;SVC&lt;/th&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row8_col0&quot; class=&quot;data row8 col0&quot; &gt;0.685450&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row8_col1&quot; class=&quot;data row8 col1&quot; &gt;0.705323&lt;/td&gt;\n                        &lt;td id=&quot;T_8af3e398_ef64_11e9_a5fd_00155d01c508row8_col2&quot; class=&quot;data row8 col2&quot; &gt;0.695245&lt;/td&gt;\n            &lt;/tr&gt;\n    &lt;/tbody&gt;&lt;/table&gt;"},"id":"1006","type":"HTML"},{"attributes":{"children":[{"id":"1011","type":"HTML"}],"margin":[0,0,0,0],"name":"Row00019"},"id":"1010","type":"Row"},{"attributes":{"margin":[5,5,5,5],"name":"Markdown00014","text":"<H4>Poisonous (Support: 3915)</H4>"},"id":"1009","type":"Div"},{"attributes":{"align":"end","children":[{"id":"1009","type":"Div"}],"margin":[0,0,0,0],"max_height":45,"name":"Row00016"},"id":"1008","type":"Row"},{"attributes":{"children":[{"id":"1002","type":"Column"},{"id":"1007","type":"Column"}],"margin":[0,0,0,0],"name":"Row00006"},"id":"1001","type":"Row"},{"attributes":{"children":[{"id":"1006","type":"HTML"}],"margin":[0,0,0,0],"name":"Row00012"},"id":"1005","type":"Row"},{"attributes":{"children":[{"id":"1008","type":"Row"},{"id":"1010","type":"Row"}],"margin":[0,0,0,0],"name":"Column00020"},"id":"1007","type":"Column"},{"attributes":{"margin":[5,5,5,5],"name":"Markdown00007","text":"<H4>Edible (Support: 4208)</H4>"},"id":"1004","type":"Div"}],"root_ids":["1001"]},"title":"Bokeh Application","version":"1.3.4"}};
  var render_items = [{"docid":"4ef0924c-19fd-4422-8f9f-8baf420825de","roots":{"1001":"6fc9a8d6-3c6d-4ecf-b224-674bbf5941d0"}}];
  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);

  }
  if (root.Bokeh !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined) {
        embed_document(root);
        clearInterval(timer);
      }
      attempts++;
      if (attempts > 100) {
        console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        clearInterval(timer);
      }
    }, 10, root)
  }
})(window);</script>


## Bar plot output 
### &rarr; using `run_alternate_model_evaluation_report_bar()`:


```python
rpts.run_alternate_model_evaluation_report_bar(dfm_mushrooms)
```


![png](images/yellowbrick_cls_model_selection_16_0.png)


### same with x-limit set to 1:


```python
rpts.run_alternate_model_evaluation_report_bar(dfm_mushrooms, xlim_to_1=True)
```
![png](images/yellowbrick_cls_model_selection_18_0.png)


## Example with a multi-class classification


```python
# Iris dataset & same models:
X, y, labels = rpts.get_iris_data()
models = rpts.get_models()
labels

dfm_iris = rpts.get_scores_df(models, X, y, labels, encode=False)
```
    array(['setosa', 'versicolor', 'virginica'], dtype='<U10')

```python
rpts.run_alternate_model_evaluation_report_bar(dfm_iris)
```
![png](images/yellowbrick_cls_model_selection_21_0.png)

---
# Using radar plots: one plot for each class.

*I've attempted reproducing the radar plots in a single row (whenever possible), but that implementation needs more work as the plots end up being squished too close together.*


```python
def scores_radar_plot_example(dfm_iris, cat='setosa'):
    supp = dfm_iris.loc[(dfm_iris.index.get_level_values(0)[0],
                         dfm_iris.index.get_level_values(1) == cat),
                        'Support'].values[0]
    
    df_set = dfm_iris.loc[dfm_iris.index.get_level_values(1) == cat,
                          dfm_iris.columns[:-1]].reset_index(level=1, drop=True)
    df_set.index.name = '{} (support: {})'.format(cat.title(), supp)
    
    ax = rpts.scores_radar_plot(df_set)
    
    '''
    # a bit too crowded with additional text:
    fig = plt.gcf()
    s = 'Radar plot of model selection scores for class {}'.format(cat.title())
    fig.text(0.5, -0.05, s,
             ha='center',
             fontsize='large')
    '''
    plt.show();
```

I'm glad I went through adapting DeepMind/bsuite radar charts, but I am I not quite satisfied with the outcome, at least with the Iris dataset: they only make it easy to id the least performant model, here SGDClassifier.  
Additionally, until &emdash;I find a way to line up the plots more compactly, they also suffer from the same 'scrolling objection' I initially made...with only 3 classes!


```python
for lbl in labels:
    scores_radar_plot_example(dfm_iris, cat=lbl)
```

![png](images/yellowbrick_cls_model_selection_25_0.png)

![png](images/yellowbrick_cls_model_selection_25_1.png)

![png](images/yellowbrick_cls_model_selection_25_2.png)
