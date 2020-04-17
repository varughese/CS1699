<body>
  <h2>CS 1699: Homework 5 </h2>
  <b>Due: </b>April 16, 11:59 PM (EST)
  <br><br>

  This assignment is worth 50 points. Please contact Mingda Zhang (mzhang@cs.pitt.edu) if you have any issues/questions regarding this assignment.<br>
  Before you start, we provided the starter code for this assignment <a href="hw5_starter.zip">here</a>.
  We <b>strongly recommend</b> you to spend some time to read the recommended implementation in starter code.<br>
  Excluding the time for training the models (please leave a few days for training), we expect this assignment to take no more than 12 hours.<br>

  <span style="color: red;">Updates: You are not allowed to use the native <a href="https://pytorch.org/docs/stable/nn.html#lstmcell" target="_blank">torch.nn.LSTMCell</a> or other built-in RNN modules in this assignment.</span>
  <br><br>

  <u>Part I: Sentiment analysis on IMDB reviews</u> (20 points)
  <br><br>
  Large Moview Review Dataset (IMDB) is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.
  It provides a set of 50,000 highly polar movie reviews. We have split the dataset into training (45,000) and test (5,000). The positive:negative ratio is 1:1 in both splits. <br>
  In this task, you need to develop a RNN model to "read" each review then predict whether it's positive or negative. <br>
  In the provided starter code, we implemented a RNN pipeline (<span class="fw">sentiment_analysis.py</span>) and a GRUCell (<span class="fw">rnn_modules.py</span>) as an example. <br>
  You need to build a few other variants of RNN modules, specifically as explained in this <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/" target="_blank">blog</a>.<br><br>

  <b>Instructions:</b>
  <ol>
    <li>(2.5 points) Read the code in <b>datasets.py</b> and <b>sentiment_analysis.py</b>, then run the training with provided GRU cell. You should be able to achieve at least 85% accuracy in 50 epochs with default hyperparameters. <br>
      Attach the figures (either TensorBoard screenshot or plot on your own) of (1) training loss, (2) training accuracy per epoch and (3) validation accuracy per epoch in your report. </li>
    <li>(4 points each) Implement the following three variants in <b>rnn_modules.py</b>, details in the <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/" target="_blank">blog</a>, the section named <i>Variants on Long Short Term Memory</i>.
      <ul>
        <li><code>LSTMCell</code>: classical LSTM with input gate and forget gate, also cell state and hidden state.</li>
        <li><code>PeepholedLSTMCell</code>: LSTM with peephole connections: adding "peephole connections" means that we let the gate layers look at the cell state.</li>
        <li><code>CoupledLSTMCell</code>: LSTM with coupled forget and input gates: instead of separately deciding what to forget and what we should add new information to, we make those decisions together.</li>
      </ul>
      Please note that the class is already provided for you, and you only need to complete the <code>__init__</code> and <code>forward</code> functions. Please do NOT change the signatures. <br>
      Finish these three classes and submit
    </li>
    <li>(2.5 points) Print number of model parameters for different module type (<code>GRUCell</code>, <code>LSTMCell</code>, <code>PeepholedLSTMCell</code> and <code>CoupledLSTMCell</code>) using <code>count_parameters</code>, and include the comparison in your report. <br>
      Use the following hyperparameters for comparison: <code>input_size=128, hidden_size=100, bias=True</code>.
    </li>
    <li>(3 points) Run experiments with your custom implementation on sentiment analysis with IMDB dataset, and compare the results with the GRU, including both speed and performance. <br>
      Attach the training loss and training/validation accuracy plot in your report.</li>
  </ol>

  <br>

  <u>Part II: Building a Shakespeare writer</u> (15 points)
  <br><br>
  RNN has demonstrated great potential in modeling language, and one interesting property is that it can "learn" to generate new sentences. <br>
  In this task, you need to develop a character-based RNN model (meaning instead of words, RNN processed one character at a time) to learn how to write like Shakespeare.
  <br><br>

  <b>Instructions:</b>
  <ol>
    <li>(4 pts) Read the code in <b>datasets.py</b> and <b>sentence_generation.py</b>, and complete the <code>SentenceGeneration</code> class which is a character-level RNN. <br>
      You can reuse (by copy/paste) most of the codes from <code>SentimentClassification</code> in Part I, just note that instead of predicting positive or negative, now your task is to predict the next character given a sequence of chars (history). <br>
    </li>
    <li>(4 pts) Train the model with the GRU module on Shakespear books. You should be able to achieve loss value of 1.2 in 10 epochs with default hyperparameters. <span style="color: red;">(Update: You probably need to use an embedding_dim of 256 and hidden_size of 512 to get the above loss value.)</span><br>
      If you are interested you could try with your own LSTM variants, but experiments with GRU is required.</li>
    <li>(7 pts) Complete the function in <b>sentence_generation.py</b> to load your trained model and generate new sentence from it. <br>
      Basically, once a language model is trained, it is able to predict the next character after a sequence, and this process can be continued (predicted character serve as history for predicting the next). <br>
      More specifically, your model should be able to predict the probability distribution over the vocabulary for the next character, and we have implemented a sampler <code>sample_next_char_id</code> which samples according to the probability. By repeating this process, your model is able to write arbitrarily long paragraphs. <br>
      For example the following passage is written by a GRU trained on Shakespeare:<br>
      <pre>ROMEO:Will't Marcius Coriolanus and envy of smelling!

DUKE VINCENTIO:
He seems muster in the shepherd's bloody winds;
Which any hand and my folder sea fast,
Last vantage doth be willing forth to have.
Sirraher comest that opposite too?

JULIET:
Are there incensed to my feet relation!
Down with mad appelate bargage! troubled
My brains loved and swifter than edwards:
Or, hency, thy fair bridging courseconce,
Or else had slept a traitors in mine own.
Look, Which canst thou have no thought appear.

ROMEO:
Give me them: for that I put us empty.

RIVERS:
The shadow doth not live: and he would not
From thee for his that office past confusion
Is their great expecteth on the wheek;
But not the noble fathom were an poison
Here come to make a dukedom: therefore--
But O, God grant! for Signior HERY

VI:
Soft love, that Lord Angelo: then blaze me all;
And slept not without a Calivan Us.
</pre>
  </ol>

  <br>
  <u>Part III: Visualization of the LSTM gates</u> (15 points)
  <br><br>
  We provided a RNN model with GRU module trained on <i>War and Peace</i> and you should visualize the learned model parameters to reveal the internal mechanism in GRU.<br>
  <ol>
    <li>(3 pts) Read the code in <b>visualization.py</b> and complete the function <code></code> to visualize the per-step, per-hidden-cell activations in your RNN using heatmap. The provided model checkpoint is in the data directory ("war_and_peace_model_checkpoint.pt"). You may reuse some visualization codes from your assignment 4. <span style="color: red;">(Updates: We have implemented the <i>model</i> for you in class <code>VisualizeInternalGates</code> and the <i>dataset</i> in <code>VisualizeWarAndPeaceDataset</code>. You can just build your model and dataset from these two classes.)</span></li>
    <li>(2 pts) Visualize the responses on the selected sentences in <code>data/war_and_peace_visualize.txt</code>, including <i>update gate</i>, <i>reset gate</i> and <i>internal cell candidates</i>.</li>
    <li>(7 pts) Modify the code to visualize different gates for the LSTM models you trained in Part I. You can reuse (but may need minor modifications) the codes in <code>VisualizeInternalGates</code> and <code>VisualizeGRUCell</code>.
    </li><li>(3 pts) Describe what you observed from the visualization. <br>
      More specifically, you should look for general patterns in the figure. For example, in the figure below for <i>update</i> gate, each row represents one hidden cell and each column represents characters in the sequence (after each character being processed). You should look at the image zoomed-out and look for thick columns that are generally more different than the surroundings. <br>
    </li>
  </ol>

  <b>Submission:</b> Please include the following files:
  <ul>
    <li>
      <span class="fw">report.pdf/docx</span>
    </li>
    <li>
      <span class="fw">
        <ul>
          <li>datasets.py</li>
          <li>visualization.py</li>
          <li>sentence_generation.py</li>
          <li>sentiment_analysis.py</li>
          <li>rnn_modules.py</li>
        </ul>
      </span>
    </li>
  </ul>
  <br>



</body>
