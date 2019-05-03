# OsCar
<p>We are OutStanding CApstoneRs (OsCar). Our main objective is to improve the existing baseline model for the <a href="https://allennlp.org/drop">DROP dataset</a>. We will do so by trying to incorporate <a href="https://arxiv.org/abs/1808.00508">NALUs</a> into our model.

<h2>Team Members</h2>
<ul>
    <li><a href="https://www.linkedin.com/in/patelr3/">Ravi Patel</a></li>
    <li><a href="https://www.linkedin.com/in/tyler-ohlsen/">Tyler Ohlsen</a></li>
    <li><a href="https://github.com/BBBBlarry">Blarry Wang</a></li>
</ul>

<h2>Goals</h2>
<ul>
    <li>Try ELMo.</li>
</ul>
<h2>Developer Notes</h2>
<ul>
    <li>Be sure to update your allennlp so that we can get the DROP dataset/models: <code>pip install allennlp --upgrade</code></li>
    <li>We are following <a href="https://github.com/allenai/allennlp-as-a-library-example/tree/master">this library structure</a>. Please be sure to follow this structure so we can maintain organization. Properly name your files and output folders.</li>
    <li>Do not add models (*.th) to the repo.</li>
    <li>The original baseline configuration was referenced from <a href="https://github.com/allenai/allennlp/blob/master/training_config/naqanet.jsonnet">here</a>.
    <li>Yes. The model takes forever to train.</li>
    <li>To run the training script, do it from the top-level of the repo: <code>./scripts/train.sh experiments/baseline.jsonnet out/test</code>. First argument is the config file. Second is the output file. Will automatically run in background so you can log off if needed. Will also automatically REMOVE THE SPECIFIED OUTPUT DIRECTORY. BE CAREFUL.</li>
</ul>
